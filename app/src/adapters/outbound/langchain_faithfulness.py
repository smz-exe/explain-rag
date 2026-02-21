import asyncio
import json
import logging
import re

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import ClaimVerification, FaithfulnessResult
from src.domain.ports.faithfulness import FaithfulnessPort, FaithfulnessVerificationError

logger = logging.getLogger(__name__)


DECOMPOSE_PROMPT = """Decompose the following answer into individual factual claims.
Return a JSON array of strings, each being one distinct claim.

Answer:
{answer}

Output only the JSON array, no other text:"""


VERIFY_PROMPT = """You are a faithfulness evaluator. Determine if each claim is supported by the provided context chunks.

Claims to verify:
{claims}

Context chunks:
{chunks}

For EACH claim, evaluate whether it is supported by the chunks. Respond with a JSON array where each element has:
{{
    "claim_index": <0-based index of the claim>,
    "verdict": "supported" or "unsupported" or "partial",
    "evidence_chunk_indices": [list of 1-based chunk numbers that support/refute the claim],
    "reasoning": "brief explanation of your verdict"
}}

Output only the JSON array, no other text:"""


class LangChainFaithfulness(FaithfulnessPort):
    """Faithfulness verification adapter using LangChain."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        max_tokens: int = 2048,
    ):
        """Initialize the faithfulness adapter.

        Args:
            model: Anthropic model name.
            api_key: Anthropic API key.
            max_tokens: Maximum tokens in response.
        """
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._llm: ChatAnthropic | None = None

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy-load the LLM client."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self._model,
                api_key=self._api_key,
                max_tokens=self._max_tokens,
                temperature=0.0,  # Deterministic for evaluation
            )
        return self._llm

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            )
        ),
        reraise=True,
    )
    def _invoke_with_retry(self, messages):
        """Invoke LLM with retry logic for transient errors."""
        return self.llm.invoke(messages)

    async def verify(
        self,
        answer: str,
        chunks: list[Chunk],
    ) -> FaithfulnessResult:
        """Verify faithfulness by decomposing answer and checking all claims in batch."""
        try:
            # Step 1: Decompose answer into claims
            logger.debug("Decomposing answer into claims...")
            claims = await self._decompose_answer(answer)

            if not claims:
                # No claims to verify
                return FaithfulnessResult(score=1.0, claims=[])

            # Step 2: Verify all claims in a single batch call
            logger.debug(f"Verifying {len(claims)} claims in batch...")
            claim_results = await self._verify_claims_batch(claims, chunks)

            # Step 3: Calculate overall score
            score = self._calculate_score(claim_results)

            return FaithfulnessResult(
                score=score,
                claims=claim_results,
            )

        except Exception as e:
            logger.error(f"Faithfulness verification failed: {e}")
            raise FaithfulnessVerificationError(f"Failed to verify faithfulness: {e}") from e

    async def _decompose_answer(self, answer: str) -> list[str]:
        """Decompose answer into individual claims."""
        prompt = DECOMPOSE_PROMPT.format(answer=answer)

        response = await asyncio.to_thread(
            self._invoke_with_retry, [HumanMessage(content=prompt)]
        )

        # Parse JSON response
        try:
            content = response.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            claims = json.loads(content)
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse claims JSON: {response.content}")
            # Fallback: split by sentences
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]

    async def _verify_claims_batch(
        self, claims: list[str], chunks: list[Chunk]
    ) -> list[ClaimVerification]:
        """Verify all claims in a single LLM call (batched)."""
        # Format claims with indices
        claims_text = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims))
        chunks_text = self._format_chunks(chunks)
        prompt = VERIFY_PROMPT.format(claims=claims_text, chunks=chunks_text)

        response = await asyncio.to_thread(
            self._invoke_with_retry, [HumanMessage(content=prompt)]
        )

        # Parse JSON array response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            results = json.loads(content)

            if not isinstance(results, list):
                results = [results]

            # Build ClaimVerification objects, indexed by claim_index
            verifications = []
            results_by_index = {r.get("claim_index", i): r for i, r in enumerate(results)}

            for i, claim in enumerate(claims):
                result = results_by_index.get(i, {})

                # Map chunk indices to IDs
                evidence_ids = []
                for idx in result.get("evidence_chunk_indices", []):
                    if isinstance(idx, int) and 1 <= idx <= len(chunks):
                        evidence_ids.append(chunks[idx - 1].id)

                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        verdict=result.get("verdict", "unsupported"),
                        evidence_chunk_ids=evidence_ids,
                        reasoning=result.get("reasoning", ""),
                    )
                )

            return verifications

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse batch verification JSON: {response.content}")
            # Fallback: mark all as unsupported
            return [
                ClaimVerification(
                    claim=claim,
                    verdict="unsupported",
                    evidence_chunk_ids=[],
                    reasoning="Failed to parse verification response",
                )
                for claim in claims
            ]

    def _format_chunks(self, chunks: list[Chunk]) -> str:
        """Format chunks with numbers for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(f"Chunk [{i}]:\n{chunk.content}\n")
        return "\n".join(formatted)

    def _calculate_score(self, results: list[ClaimVerification]) -> float:
        """Calculate overall faithfulness score."""
        if not results:
            return 1.0

        # Score: supported=1.0, partial=0.5, unsupported=0.0
        verdict_scores = {
            "supported": 1.0,
            "partial": 0.5,
            "unsupported": 0.0,
        }

        total = sum(verdict_scores.get(r.verdict, 0.0) for r in results)
        return total / len(results)

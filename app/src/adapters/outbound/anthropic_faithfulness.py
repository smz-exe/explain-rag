"""Faithfulness verification adapter calling the Anthropic SDK directly."""

import json
import logging
import re

from anthropic import AsyncAnthropic

from src.adapters.outbound.anthropic_client import (
    build_client,
    parse_json_response,
    response_text,
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
    "reasoning": "one short sentence explaining the verdict"
}}

Output only the JSON array, no other text:"""

# Claims per verification call. A batch of 12 with one-sentence reasoning
# stays comfortably inside the response token budget; unbatched 30+ claim
# responses were truncated at max_tokens in production, which the JSON
# parser then rejected wholesale (0% faithfulness on faithful answers).
VERIFY_BATCH_SIZE = 12
# Response budget per claim (JSON overhead + reasoning), plus fixed headroom
VERIFY_TOKENS_PER_CLAIM = 200
VERIFY_TOKENS_HEADROOM = 500
VERIFY_MAX_TOKENS_CEILING = 8192


class AnthropicFaithfulness(FaithfulnessPort):
    """Faithfulness verification adapter using the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str = "",
        max_tokens: int = 2048,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: AsyncAnthropic | None = None,
    ):
        """Initialize the faithfulness adapter.

        Args:
            model: Anthropic model name.
            api_key: Anthropic API key.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: SDK retry count for transient failures.
            client: Optional pre-built client (used in tests).
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or build_client(api_key, timeout, max_retries)

    async def verify(
        self,
        answer: str,
        chunks: list[Chunk],
    ) -> FaithfulnessResult:
        """Verify faithfulness by decomposing answer and checking all claims in batch."""
        try:
            logger.debug("Decomposing answer into claims...")
            claims = await self._decompose_answer(answer)

            if not claims:
                return FaithfulnessResult(score=1.0, claims=[])

            logger.debug(f"Verifying {len(claims)} claims in batch...")
            claim_results = await self._verify_claims_batch(claims, chunks)

            score = self._calculate_score(claim_results)

            return FaithfulnessResult(
                score=score,
                claims=claim_results,
            )

        except Exception as e:
            logger.error(f"Faithfulness verification failed: {e}")
            raise FaithfulnessVerificationError(f"Failed to verify faithfulness: {e}") from e

    async def _complete(self, prompt: str, max_tokens: int | None = None) -> str:
        """Run a single user-prompt completion and return its text."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning("Faithfulness completion truncated at max_tokens — output incomplete")
        return response_text(response)

    async def _decompose_answer(self, answer: str) -> list[str]:
        """Decompose answer into individual claims."""
        content = await self._complete(DECOMPOSE_PROMPT.format(answer=answer))

        try:
            claims = parse_json_response(content)
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse claims JSON: {content}")
            # Fallback: split by sentences
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]

    async def _verify_claims_batch(
        self, claims: list[str], chunks: list[Chunk]
    ) -> list[ClaimVerification]:
        """Verify claims in batches sized to keep responses under the token budget."""
        verifications: list[ClaimVerification] = []
        for start in range(0, len(claims), VERIFY_BATCH_SIZE):
            batch = claims[start : start + VERIFY_BATCH_SIZE]
            verifications.extend(await self._verify_batch(batch, chunks))
        return verifications

    async def _verify_batch(
        self, claims: list[str], chunks: list[Chunk]
    ) -> list[ClaimVerification]:
        """Verify one batch of claims in a single LLM call."""
        claims_text = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims))
        chunks_text = self._format_chunks(chunks)
        max_tokens = min(
            VERIFY_MAX_TOKENS_CEILING,
            max(self._max_tokens, VERIFY_TOKENS_PER_CLAIM * len(claims) + VERIFY_TOKENS_HEADROOM),
        )
        content = await self._complete(
            VERIFY_PROMPT.format(claims=claims_text, chunks=chunks_text), max_tokens=max_tokens
        )

        try:
            results = parse_json_response(content)

            if not isinstance(results, list):
                results = [results]

            verifications = []
            results_by_index = {r.get("claim_index", i): r for i, r in enumerate(results)}

            for i, claim in enumerate(claims):
                result = results_by_index.get(i, {})

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
            logger.warning(f"Failed to parse batch verification JSON: {content}")
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

"""RAG generation adapter calling the Anthropic SDK directly."""

import logging
import re

from anthropic import AsyncAnthropic

from src.adapters.outbound.anthropic_client import build_client, response_text
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Citation, GenerationResult
from src.domain.ports.llm import InsufficientContextError, LLMGenerationError, LLMPort

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context chunks.

CITATION RULES:
1. Use bracket notation [1], [2], etc. to cite sources inline
2. The number corresponds to the chunk's position (Chunk [1], Chunk [2], etc.)
3. Every factual claim MUST have at least one citation
4. If you cannot answer from the provided context, respond with exactly: "I cannot answer this question based on the available context."
5. Do not make up information not present in the chunks
6. Place citations immediately after the relevant claim
7. Multiple citations can be combined: [1][2]

Provide a clear, concise answer with inline citations."""

CONTEXT_TEMPLATE = """Context chunks:

{chunks}

Question: {question}

Answer (with inline citations):"""


class AnthropicRAG(LLMPort):
    """LLM adapter using the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str = "",
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: AsyncAnthropic | None = None,
    ):
        """Initialize the RAG adapter.

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

    async def generate(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> GenerationResult:
        """Generate an answer with inline citations."""
        if not chunks:
            raise InsufficientContextError("No chunks provided for context")

        user_prompt = CONTEXT_TEMPLATE.format(
            chunks=self._format_chunks(chunks),
            question=question,
        )

        try:
            logger.debug(f"Generating answer for: {question[:50]}...")
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_answer = response_text(response)

            # Check for insufficient context response
            if "cannot answer" in raw_answer.lower() and "available context" in raw_answer.lower():
                raise InsufficientContextError(raw_answer)

            citations = self._extract_citations(raw_answer, chunks)

            return GenerationResult(
                answer=raw_answer,
                citations=citations,
                raw_response=raw_answer,
            )

        except InsufficientContextError:
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate answer: {e}") from e

    def _format_chunks(self, chunks: list[Chunk]) -> str:
        """Format chunks with rank numbers for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            paper_title = chunk.metadata.get("paper_title", "Unknown")
            formatted.append(f"Chunk [{i}] (Paper: {paper_title}):\n{chunk.content}\n")
        return "\n".join(formatted)

    def _extract_citations(self, answer: str, chunks: list[Chunk]) -> list[Citation]:
        """Extract citation mappings from the answer text.

        Parses bracket notation [1], [2], etc. and maps to chunk IDs.
        Groups citations by the sentence/claim they appear in.
        """
        citations = []

        sentences = re.split(r"(?<=[.!?])\s+", answer)

        for sentence in sentences:
            citation_matches = re.findall(r"\[(\d+)\]", sentence)
            if citation_matches:
                chunk_indices = sorted(set(int(m) for m in citation_matches))

                chunk_ids = []
                for idx in chunk_indices:
                    if 1 <= idx <= len(chunks):
                        chunk_ids.append(chunks[idx - 1].id)

                if chunk_ids:
                    claim = re.sub(r"\[\d+\]", "", sentence).strip()
                    if claim:
                        citations.append(
                            Citation(
                                claim=claim,
                                chunk_ids=chunk_ids,
                                confidence=0.9,  # Default confidence
                            )
                        )

        return citations

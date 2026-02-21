import asyncio
import logging
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

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


class LangChainRAG(LLMPort):
    """LLM adapter using LangChain with ChatAnthropic."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        """Initialize the LangChain RAG adapter.

        Args:
            model: Anthropic model name.
            api_key: Anthropic API key.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 for deterministic).
        """
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._llm: ChatAnthropic | None = None

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy-load the LLM client."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self._model,
                api_key=self._api_key,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        return self._llm

    async def generate(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> GenerationResult:
        """Generate an answer with inline citations."""
        if not chunks:
            raise InsufficientContextError("No chunks provided for context")

        # Format chunks with rank numbers
        chunks_text = self._format_chunks(chunks)

        # Build prompt
        user_prompt = CONTEXT_TEMPLATE.format(
            chunks=chunks_text,
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Run LLM in thread to avoid blocking
            logger.debug(f"Generating answer for: {question[:50]}...")
            response = await asyncio.to_thread(self.llm.invoke, messages)
            raw_answer = response.content

            # Check for insufficient context response
            if "cannot answer" in raw_answer.lower() and "available context" in raw_answer.lower():
                raise InsufficientContextError(raw_answer)

            # Extract citations from the answer
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

        # Split answer into sentences
        sentences = re.split(r"(?<=[.!?])\s+", answer)

        for sentence in sentences:
            # Find all bracket citations in this sentence
            citation_matches = re.findall(r"\[(\d+)\]", sentence)
            if citation_matches:
                # Get unique chunk references
                chunk_indices = sorted(set(int(m) for m in citation_matches))

                # Map to chunk IDs
                chunk_ids = []
                for idx in chunk_indices:
                    if 1 <= idx <= len(chunks):
                        chunk_ids.append(chunks[idx - 1].id)

                if chunk_ids:
                    # Remove citation markers for the claim text
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

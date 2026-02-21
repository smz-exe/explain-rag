from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import GenerationResult


class LLMPort(ABC):
    """Abstract interface for LLM-based answer generation."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> GenerationResult:
        """Generate an answer grounded in the provided chunks.

        Args:
            question: The user's natural language question.
            chunks: Retrieved chunks to use as context (ordered by rank).

        Returns:
            GenerationResult with answer text and citation mappings.

        Raises:
            LLMGenerationError: If generation fails.
            InsufficientContextError: If context is insufficient to answer.
        """
        ...


class LLMGenerationError(Exception):
    """Raised when LLM generation fails."""

    pass


class InsufficientContextError(Exception):
    """Raised when there is insufficient context to answer the question."""

    pass

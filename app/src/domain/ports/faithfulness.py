from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import FaithfulnessResult


class FaithfulnessPort(ABC):
    """Abstract interface for answer faithfulness verification."""

    @abstractmethod
    async def verify(
        self,
        answer: str,
        chunks: list[Chunk],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness of the answer against source chunks.

        Args:
            answer: The generated answer to verify.
            chunks: The source chunks used to generate the answer.

        Returns:
            FaithfulnessResult with overall score and per-claim verdicts.

        Raises:
            FaithfulnessVerificationError: If verification fails.
        """
        ...


class FaithfulnessVerificationError(Exception):
    """Raised when faithfulness verification fails."""

    pass

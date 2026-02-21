"""Reranker port for cross-encoder reranking."""

from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk


class RerankerPort(ABC):
    """Abstract interface for reranking retrieved chunks."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks by relevance to the query.

        Args:
            query: The user's question.
            chunks: Chunks to rerank.
            top_k: Optional limit on returned chunks. If None, return all.

        Returns:
            List of (chunk, score) tuples sorted by relevance descending.
        """
        ...

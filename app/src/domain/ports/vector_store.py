from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk


class VectorStorePort(ABC):
    """Abstract interface for vector storage and retrieval operations."""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their corresponding embeddings.

        Args:
            chunks: List of Chunk entities to store.
            embeddings: List of embedding vectors, one per chunk.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding vector.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (e.g., {"paper_id": "..."}).

        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance.
        """
        ...

    @abstractmethod
    async def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with stats like chunk_count, paper_count, etc.
        """
        ...

    @abstractmethod
    async def list_papers(self) -> list[dict]:
        """List all papers that have chunks in the store.

        Returns:
            List of paper metadata dictionaries.
        """
        ...

    @abstractmethod
    async def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a given paper.

        Args:
            paper_id: The paper ID to delete chunks for.

        Returns:
            Number of chunks deleted.
        """
        ...

    @abstractmethod
    async def get_paper_embeddings(self) -> list[tuple[str, list[float]]]:
        """Get mean embedding for each paper.

        Computes the mean of all chunk embeddings for each paper.

        Returns:
            List of (paper_id, mean_embedding) tuples.
        """
        ...

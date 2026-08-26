from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.paper import Paper


class VectorStorePort(ABC):
    """Abstract interface for vector storage and retrieval operations."""

    @abstractmethod
    async def add_chunks(
        self, paper: Paper, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Store a paper together with its chunks and their embeddings.

        The paper is passed explicitly rather than reconstructed from chunk
        metadata: an implementation must not have to guess which free-form
        metadata keys carry paper-level fields.

        Args:
            paper: The paper these chunks belong to.
            chunks: List of Chunk entities to store.
            embeddings: List of embedding vectors, one per chunk.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        paper_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding vector.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            paper_ids: Optional list of paper IDs to restrict the search to.
                None searches every paper. Scoping is applied before top_k, so
                a scoped search still returns up to top_k matching chunks.

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
    async def delete_paper(self, paper_id: str) -> int | None:
        """Delete a paper and its chunks.

        Args:
            paper_id: The paper ID to delete.

        Returns:
            Number of chunks deleted, or None if no such paper existed. A
            deleted paper that happened to have no chunks returns 0, which is
            not the same answer as "not found".
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

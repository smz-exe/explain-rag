from abc import ABC, abstractmethod


class ClusteringPort(ABC):
    """Abstract interface for clustering operations."""

    @abstractmethod
    async def cluster(
        self,
        embeddings: list[list[float]],
    ) -> list[int]:
        """Assign cluster labels to embeddings.

        Args:
            embeddings: List of embedding vectors to cluster.

        Returns:
            List of cluster labels, one per embedding.
            -1 indicates noise/outlier (not assigned to any cluster).
        """
        ...

    @abstractmethod
    async def get_cluster_count(self) -> int:
        """Get the number of clusters found (excluding noise).

        Returns:
            Number of distinct clusters.

        Raises:
            RuntimeError: If called before cluster().
        """
        ...

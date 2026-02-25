"""HDBSCAN clustering adapter."""

import asyncio

import hdbscan
import numpy as np

from src.domain.ports.clustering import ClusteringPort


class HDBSCANClusterer(ClusteringPort):
    """Clustering adapter using HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ):
        """Initialize the HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum number of samples in a cluster.
            min_samples: Number of samples in a neighborhood for core points.
            metric: Distance metric to use.
            cluster_selection_method: Method for cluster selection ('eom' or 'leaf').
        """
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._metric = metric
        self._cluster_selection_method = cluster_selection_method
        self._labels: np.ndarray | None = None
        self._cluster_count: int = 0

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
        if not embeddings:
            self._labels = np.array([])
            self._cluster_count = 0
            return []

        # Adjust min_cluster_size if we have very few samples
        min_cluster_size = min(self._min_cluster_size, len(embeddings))
        min_cluster_size = max(min_cluster_size, 2)  # HDBSCAN needs at least 2

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self._min_samples,
            metric=self._metric,
            cluster_selection_method=self._cluster_selection_method,
        )

        embeddings_array = np.array(embeddings)
        self._labels = await asyncio.to_thread(clusterer.fit_predict, embeddings_array)

        # Count unique clusters (excluding -1 which is noise)
        unique_labels = set(self._labels.tolist())
        unique_labels.discard(-1)
        self._cluster_count = len(unique_labels)

        return self._labels.tolist()

    async def get_cluster_count(self) -> int:
        """Get the number of clusters found (excluding noise).

        Returns:
            Number of distinct clusters.

        Raises:
            RuntimeError: If called before cluster().
        """
        if self._labels is None:
            raise RuntimeError("Clustering has not been performed. Call cluster() first.")

        return self._cluster_count

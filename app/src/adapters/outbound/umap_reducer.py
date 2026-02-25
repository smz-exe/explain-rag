"""UMAP dimensionality reduction adapter."""

import asyncio

import numpy as np
import umap

from src.domain.ports.dimensionality_reduction import DimensionalityReductionPort


class UMAPReducer(DimensionalityReductionPort):
    """Dimensionality reduction adapter using UMAP."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        """Initialize the UMAP reducer.

        Args:
            n_neighbors: Number of neighbors to consider for manifold approximation.
            min_dist: Minimum distance between points in low-dimensional space.
            metric: Distance metric to use (cosine works well for embeddings).
            random_state: Random seed for reproducibility.
        """
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._metric = metric
        self._random_state = random_state
        self._reducer: umap.UMAP | None = None

    def is_fitted(self) -> bool:
        """Check if the reducer has been fitted."""
        return self._reducer is not None and hasattr(self._reducer, "embedding_")

    async def fit_transform(
        self,
        embeddings: list[list[float]],
        n_components: int = 3,
    ) -> list[tuple[float, float, float]]:
        """Fit UMAP and transform embeddings to lower dimensions.

        Args:
            embeddings: List of high-dimensional embedding vectors.
            n_components: Target dimensionality (default 3 for 3D visualization).

        Returns:
            List of coordinate tuples, one per input embedding.
        """
        if not embeddings:
            return []

        # Adjust n_neighbors if we have fewer samples
        n_neighbors = min(self._n_neighbors, len(embeddings) - 1)
        n_neighbors = max(n_neighbors, 2)  # UMAP needs at least 2 neighbors

        self._reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self._min_dist,
            metric=self._metric,
            random_state=self._random_state,
        )

        embeddings_array = np.array(embeddings)
        result = await asyncio.to_thread(self._reducer.fit_transform, embeddings_array)

        return [tuple(row.tolist()) for row in result]

    async def transform(
        self,
        embeddings: list[list[float]],
    ) -> list[tuple[float, float, float]]:
        """Transform new embeddings using the fitted UMAP model.

        Args:
            embeddings: List of high-dimensional embedding vectors.

        Returns:
            List of coordinate tuples in the same space as fit_transform output.

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self.is_fitted():
            raise RuntimeError("UMAP reducer has not been fitted. Call fit_transform first.")

        if not embeddings:
            return []

        embeddings_array = np.array(embeddings)
        result = await asyncio.to_thread(self._reducer.transform, embeddings_array)

        return [tuple(row.tolist()) for row in result]

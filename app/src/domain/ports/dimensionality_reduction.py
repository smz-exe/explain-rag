from abc import ABC, abstractmethod


class DimensionalityReductionPort(ABC):
    """Abstract interface for dimensionality reduction operations."""

    @abstractmethod
    async def fit_transform(
        self,
        embeddings: list[list[float]],
        n_components: int = 3,
    ) -> list[tuple[float, float, float]]:
        """Fit the reducer and transform embeddings to lower dimensions.

        Args:
            embeddings: List of high-dimensional embedding vectors.
            n_components: Target dimensionality (default 3 for 3D visualization).

        Returns:
            List of coordinate tuples, one per input embedding.
        """
        ...

    @abstractmethod
    async def transform(
        self,
        embeddings: list[list[float]],
    ) -> list[tuple[float, float, float]]:
        """Transform new embeddings using an already-fitted reducer.

        Args:
            embeddings: List of high-dimensional embedding vectors.

        Returns:
            List of coordinate tuples in the same space as fit_transform output.

        Raises:
            RuntimeError: If called before fit_transform.
        """
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the reducer has been fitted.

        Returns:
            True if fit_transform has been called, False otherwise.
        """
        ...

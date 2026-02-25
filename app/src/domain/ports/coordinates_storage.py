"""Port for coordinates persistence operations."""

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.coordinates import Cluster, PaperCoordinates


class CoordinatesStoragePort(ABC):
    """Abstract interface for coordinates persistence operations."""

    @abstractmethod
    async def load(
        self,
    ) -> tuple[list[PaperCoordinates], list[Cluster], datetime | None]:
        """Load stored coordinates and clusters.

        Returns:
            Tuple of (paper_coordinates, clusters, computed_at).
            Returns ([], [], None) if no data stored.
        """
        ...

    @abstractmethod
    async def save(
        self,
        coordinates: list[PaperCoordinates],
        clusters: list[Cluster],
        computed_at: datetime,
    ) -> None:
        """Save coordinates and clusters, replacing any existing data.

        Args:
            coordinates: List of paper coordinates to store.
            clusters: List of clusters to store.
            computed_at: Timestamp of computation.

        Note:
            If both coordinates and clusters are empty, computed_at will not
            be persisted (since it's stored per-row). Subsequent load() will
            return None for computed_at in this case.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored coordinates and clusters."""
        ...

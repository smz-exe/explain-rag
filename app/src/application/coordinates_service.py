"""Service for managing paper coordinates and clustering."""

import logging
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime

from src.domain.entities.coordinates import Cluster, PaperCoordinates
from src.domain.ports.clustering import ClusteringPort
from src.domain.ports.coordinates_storage import CoordinatesStoragePort
from src.domain.ports.dimensionality_reduction import DimensionalityReductionPort
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


class CoordinatesService:
    """Service orchestrating paper coordinate computation and clustering."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        dim_reducer: DimensionalityReductionPort,
        clusterer: ClusteringPort,
        storage: CoordinatesStoragePort | None = None,
    ):
        """Initialize the coordinates service.

        Args:
            vector_store: Adapter for retrieving paper embeddings.
            dim_reducer: Adapter for dimensionality reduction (UMAP).
            clusterer: Adapter for clustering (HDBSCAN).
            storage: Optional adapter for persisting coordinates to storage.
        """
        self._vector_store = vector_store
        self._dim_reducer = dim_reducer
        self._clusterer = clusterer
        self._storage = storage

        # In-memory cache for computed data
        self._paper_coordinates: list[PaperCoordinates] = []
        self._clusters: list[Cluster] = []
        self._computed_at: datetime | None = None

    async def initialize(self) -> None:
        """Initialize service by loading persisted data if available.

        Call this method on startup to restore coordinates from storage.
        Note: UMAP will not be fitted after loading, so get_query_coordinates()
        will return None until recompute_all() is called.
        """
        if self._storage is None:
            return

        try:
            coords, clusters, computed_at = await self._storage.load()
            if coords:
                self._paper_coordinates = coords
                self._clusters = clusters
                self._computed_at = computed_at
                logger.info(
                    f"Loaded {len(coords)} coordinates and {len(clusters)} clusters "
                    f"from storage (computed at {computed_at})"
                )
        except Exception as e:
            logger.warning(f"Failed to load coordinates from storage: {e}")

    @property
    def is_computed(self) -> bool:
        """Check if coordinates have been computed."""
        return self._computed_at is not None and len(self._paper_coordinates) > 0

    @property
    def computed_at(self) -> datetime | None:
        """Get timestamp of last computation."""
        return self._computed_at

    async def get_paper_coordinates(self) -> list[PaperCoordinates]:
        """Get cached paper coordinates.

        Returns:
            List of paper coordinates (empty if not yet computed).
        """
        return self._paper_coordinates.copy()

    async def get_clusters(self) -> list[Cluster]:
        """Get cached clusters.

        Returns:
            List of clusters (empty if not yet computed).
        """
        return self._clusters.copy()

    async def recompute_all(self) -> dict:
        """Recompute all coordinates and clusters.

        This is the main method triggered by admin to refresh the visualization data.

        Returns:
            Dictionary with computation stats.
        """
        start_time = time.perf_counter()

        logger.info("Starting coordinate recomputation...")

        # Step 1: Get all paper embeddings from vector store
        paper_embeddings = await self._vector_store.get_paper_embeddings()

        if not paper_embeddings:
            logger.warning("No paper embeddings found, clearing cache")
            self._paper_coordinates = []
            self._clusters = []
            self._computed_at = datetime.now(UTC)
            return {
                "papers_processed": 0,
                "clusters_found": 0,
                "time_ms": 0,
            }

        # Get paper metadata for titles
        papers_list = await self._vector_store.list_papers()
        paper_metadata = {p["paper_id"]: p for p in papers_list}

        paper_ids = [pid for pid, _ in paper_embeddings]
        embeddings = [emb for _, emb in paper_embeddings]

        logger.info(f"Processing {len(paper_ids)} papers")

        # Step 2: Run UMAP dimensionality reduction
        logger.debug("Running UMAP dimensionality reduction")
        coords_3d = await self._dim_reducer.fit_transform(embeddings, n_components=3)

        # Step 3: Run HDBSCAN clustering
        logger.debug("Running HDBSCAN clustering")
        cluster_labels = await self._clusterer.cluster(embeddings)
        cluster_count = await self._clusterer.get_cluster_count()

        # Step 4: Build PaperCoordinates entities (immutable list comprehension)
        self._paper_coordinates = [
            PaperCoordinates(
                paper_id=paper_id,
                arxiv_id=paper_metadata.get(paper_id, {}).get("arxiv_id", ""),
                title=paper_metadata.get(paper_id, {}).get("title", "Unknown"),
                coords=coords_3d[i],
                cluster_id=cluster_labels[i] if cluster_labels[i] >= 0 else None,
                chunk_count=paper_metadata.get(paper_id, {}).get("chunk_count", 0),
            )
            for i, paper_id in enumerate(paper_ids)
        ]

        # Step 5: Build Cluster entities with generated labels
        self._clusters = self._build_clusters(cluster_labels, paper_ids, paper_metadata)

        self._computed_at = datetime.now(UTC)

        # Persist to storage if available
        if self._storage is not None:
            await self._storage.save(
                self._paper_coordinates,
                self._clusters,
                self._computed_at,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Coordinate recomputation complete: {len(paper_ids)} papers, "
            f"{cluster_count} clusters in {elapsed_ms:.1f}ms"
        )

        return {
            "papers_processed": len(paper_ids),
            "clusters_found": cluster_count,
            "time_ms": round(elapsed_ms, 1),
        }

    def _build_clusters(
        self,
        cluster_labels: list[int],
        paper_ids: list[str],
        paper_metadata: dict[str, dict],
    ) -> list[Cluster]:
        """Build cluster entities with auto-generated labels.

        Args:
            cluster_labels: Cluster assignment for each paper (-1 for noise).
            paper_ids: List of paper IDs in order.
            paper_metadata: Metadata dict keyed by paper_id.

        Returns:
            List of Cluster entities.
        """
        # Group papers by cluster using defaultdict
        cluster_papers: dict[int, list[str]] = defaultdict(list)
        for paper_id, label in zip(paper_ids, cluster_labels, strict=True):
            if label >= 0:  # Skip noise points
                cluster_papers[label].append(paper_id)

        # Build clusters with auto-generated labels (immutable list comprehension)
        return [
            Cluster(
                id=cluster_id,
                label=self._generate_cluster_label(
                    [paper_metadata.get(pid, {}).get("title", "") for pid in pids]
                ),
                paper_ids=pids,
            )
            for cluster_id, pids in sorted(cluster_papers.items())
        ]

    def _generate_cluster_label(self, titles: list[str]) -> str:
        """Generate a cluster label from paper titles.

        Uses simple word frequency analysis to find common themes.

        Args:
            titles: List of paper titles in the cluster.

        Returns:
            A descriptive label for the cluster.
        """
        if not titles:
            return "Uncategorized"

        # Common words to exclude
        stop_words = {
            "a",
            "an",
            "the",
            "of",
            "for",
            "and",
            "in",
            "on",
            "to",
            "with",
            "is",
            "are",
            "by",
            "from",
            "as",
            "at",
            "or",
            "be",
            "this",
            "that",
            "via",
            "using",
            "based",
            "towards",
            "through",
        }

        # Extract and count significant words
        word_counts: Counter[str] = Counter()
        for title in titles:
            words = title.lower().split()
            for word in words:
                # Clean word and filter
                clean_word = "".join(c for c in word if c.isalnum())
                if clean_word and len(clean_word) > 2 and clean_word not in stop_words:
                    word_counts[clean_word] += 1

        if not word_counts:
            return "Research Papers"

        # Get top words that appear in multiple titles (or most frequent if single paper)
        threshold = max(1, len(titles) // 2)
        common_words = [word for word, count in word_counts.most_common(5) if count >= threshold]

        if not common_words:
            # Fall back to most frequent words
            common_words = [word for word, _ in word_counts.most_common(3)]

        # Title case the label
        label = " & ".join(word.title() for word in common_words[:3])

        return label or "Research Papers"

    async def get_query_coordinates(
        self,
        query_embedding: list[float],
    ) -> tuple[float, float, float] | None:
        """Project a query embedding into the existing coordinate space.

        Args:
            query_embedding: The embedding vector for the query.

        Returns:
            3D coordinates for the query, or None if not fitted.
        """
        if not self._dim_reducer.is_fitted():
            logger.warning("Dimensionality reducer not fitted, cannot project query")
            return None

        try:
            coords = await self._dim_reducer.transform([query_embedding])
            return coords[0] if coords else None
        except RuntimeError as e:
            logger.error(f"Failed to project query coordinates: {e}")
            return None

    async def clear_cache(self) -> None:
        """Clear the cached coordinates and clusters.

        Also clears storage if available.
        """
        self._paper_coordinates = []
        self._clusters = []
        self._computed_at = None

        if self._storage is not None:
            await self._storage.clear()

        logger.info("Coordinates cache cleared")

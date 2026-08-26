"""Service reporting aggregate counts across the ingested corpus."""

from dataclasses import dataclass

from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.vector_store import VectorStorePort


@dataclass(frozen=True)
class CorpusStats:
    """How much the system currently holds."""

    papers_count: int
    chunks_count: int
    queries_count: int


class StatsService:
    """Service composing corpus statistics from the stores that hold them."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        query_storage: QueryStoragePort,
    ):
        """Initialize the stats service.

        Args:
            vector_store: Adapter holding papers and chunks.
            query_storage: Adapter holding answered queries.
        """
        self._vector_store = vector_store
        self._query_storage = query_storage

    async def collect_stats(self) -> CorpusStats:
        """Collect the counts describing the corpus.

        Returns:
            Paper, chunk, and query counts.
        """
        store_stats = await self._vector_store.get_stats()
        queries_count = await self._query_storage.count()

        return CorpusStats(
            papers_count=store_stats.paper_count,
            chunks_count=store_stats.chunk_count,
            queries_count=queries_count,
        )

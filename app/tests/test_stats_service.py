"""Tests for StatsService."""

import pytest

from src.application.stats_service import CorpusStats, StatsService
from src.domain.entities.paper import StoreStats
from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.vector_store import VectorStorePort


class StubVectorStore(VectorStorePort):
    """Vector store returning fixed counts, or raising on demand."""

    def __init__(self, stats: StoreStats | None = None, error: Exception | None = None):
        self._stats = stats or StoreStats(chunk_count=0, paper_count=0)
        self._error = error

    async def get_stats(self) -> StoreStats:
        if self._error is not None:
            raise self._error
        return self._stats

    async def add_chunks(self, paper, chunks, embeddings) -> None: ...

    async def search(self, query_embedding, top_k=10, paper_ids=None):
        return []

    async def list_papers(self):
        return []

    async def delete_paper(self, paper_id):
        return None

    async def get_paper_embeddings(self):
        return []


class StubQueryStorage(QueryStoragePort):
    """Query storage reporting a fixed count."""

    def __init__(self, count: int = 0):
        self._count = count

    async def count(self) -> int:
        return self._count

    async def store(self, response) -> None: ...

    async def update(self, response) -> bool:
        return True

    async def get(self, query_id):
        return None

    async def list_recent(self, limit=20):
        return []

    async def delete(self, query_id) -> bool:
        return True

    async def list_by_verification_status(self, status: str) -> list[str]:
        return []


@pytest.mark.asyncio
class TestCollectStats:
    """Composing the corpus counts from both stores."""

    async def test_combines_counts_from_both_ports(self):
        service = StatsService(
            vector_store=StubVectorStore(StoreStats(chunk_count=4622, paper_count=28)),
            query_storage=StubQueryStorage(count=137),
        )

        stats = await service.collect_stats()

        assert stats == CorpusStats(papers_count=28, chunks_count=4622, queries_count=137)

    async def test_empty_corpus_reports_zeros(self):
        service = StatsService(
            vector_store=StubVectorStore(StoreStats(chunk_count=0, paper_count=0)),
            query_storage=StubQueryStorage(count=0),
        )

        stats = await service.collect_stats()

        assert stats == CorpusStats(papers_count=0, chunks_count=0, queries_count=0)

    async def test_store_failure_propagates(self):
        """A failing store must surface, not be reported as a corpus of zero."""
        service = StatsService(
            vector_store=StubVectorStore(error=RuntimeError("database unreachable")),
            query_storage=StubQueryStorage(count=5),
        )

        with pytest.raises(RuntimeError, match="database unreachable"):
            await service.collect_stats()

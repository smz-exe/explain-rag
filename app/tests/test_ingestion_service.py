"""Unit tests for IngestionService using in-memory port fakes."""

import asyncio

import pytest

from src.application.ingestion_service import IngestionService
from src.domain.entities.chunk import Chunk
from src.domain.entities.paper import Paper
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.paper_source import (
    PaperNotFoundError,
    PaperSourcePort,
    PDFParsingError,
)
from src.domain.ports.vector_store import VectorStorePort


class FakePaperSource(PaperSourcePort):
    """In-memory paper source with configurable per-ID failures."""

    def __init__(
        self,
        paper: Paper,
        chunks: list[Chunk],
        fetch_errors: dict[str, Exception] | None = None,
        extract_error: Exception | None = None,
        fetch_gate: asyncio.Event | None = None,
    ):
        self._paper = paper
        self._chunks = chunks
        self._fetch_errors = fetch_errors or {}
        self._extract_error = extract_error
        self._fetch_gate = fetch_gate

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        if self._fetch_gate is not None:
            await self._fetch_gate.wait()
        if arxiv_id in self._fetch_errors:
            raise self._fetch_errors[arxiv_id]
        return self._paper

    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        return [self._paper][:max_results]

    async def extract_chunks(
        self, paper: Paper, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        if self._extract_error is not None:
            raise self._extract_error
        return self._chunks


class FakeEmbedding(EmbeddingPort):
    """Deterministic embedding fake."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class RecordingVectorStore(VectorStorePort):
    """Vector store fake that records add_chunks calls."""

    def __init__(self):
        self.added_papers: list[Paper] = []
        self.added_chunks: list[Chunk] = []
        self.added_embeddings: list[list[float]] = []

    async def add_chunks(
        self, paper: Paper, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        self.added_papers.append(paper)
        self.added_chunks.extend(chunks)
        self.added_embeddings.extend(embeddings)

    async def search(
        self, query_embedding: list[float], top_k: int = 5, paper_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        return []

    async def get_stats(self) -> dict:
        return {"papers_count": 0, "chunks_count": len(self.added_chunks)}

    async def list_papers(self) -> list[dict]:
        return []

    async def delete_paper(self, paper_id: str) -> int:
        return 0

    async def get_paper_embeddings(self) -> list[tuple[str, list[float]]]:
        return []


@pytest.fixture
def vector_store() -> RecordingVectorStore:
    return RecordingVectorStore()


def make_service(
    paper_source: PaperSourcePort, vector_store: RecordingVectorStore
) -> IngestionService:
    return IngestionService(
        paper_source=paper_source,
        embedding=FakeEmbedding(),
        vector_store=vector_store,
    )


class TestIngestPaper:
    """Tests for IngestionService.ingest_paper."""

    async def test_success_stores_chunks_with_embeddings(
        self, sample_paper, sample_chunks, vector_store
    ):
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        result = await service.ingest_paper(sample_paper.arxiv_id)

        assert result.status == "success"
        assert result.arxiv_id == sample_paper.arxiv_id
        assert result.title == sample_paper.title
        assert result.chunk_count == len(sample_chunks)
        assert len(vector_store.added_chunks) == len(sample_chunks)
        assert len(vector_store.added_embeddings) == len(sample_chunks)

    async def test_success_enriches_chunk_metadata(self, sample_paper, sample_chunks, vector_store):
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        await service.ingest_paper(sample_paper.arxiv_id)

        for chunk in vector_store.added_chunks:
            assert chunk.metadata["arxiv_id"] == sample_paper.arxiv_id
            assert chunk.metadata["paper_title"] == sample_paper.title

    async def test_paper_metadata_reaches_the_store(
        self, sample_paper, sample_chunks, vector_store
    ):
        """The store needs the whole Paper, not two keys smuggled via chunk metadata.

        Only arxiv_id and paper_title were ever attached to chunks, so authors,
        abstract, url, and pdf_url were silently dropped for every paper.
        """
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        await service.ingest_paper(sample_paper.arxiv_id)

        assert vector_store.added_papers == [sample_paper]
        stored = vector_store.added_papers[0]
        assert stored.authors == sample_paper.authors
        assert stored.abstract == sample_paper.abstract
        assert stored.url == sample_paper.url
        assert stored.pdf_url == sample_paper.pdf_url

    async def test_empty_chunks_returns_error(self, sample_paper, vector_store):
        source = FakePaperSource(sample_paper, chunks=[])
        service = make_service(source, vector_store)

        result = await service.ingest_paper(sample_paper.arxiv_id)

        assert result.status == "error"
        assert result.error == "No chunks extracted from PDF"
        assert vector_store.added_chunks == []

    async def test_paper_not_found_returns_error(self, sample_paper, sample_chunks, vector_store):
        source = FakePaperSource(
            sample_paper,
            sample_chunks,
            fetch_errors={"missing.00001": PaperNotFoundError("Paper missing.00001 not found")},
        )
        service = make_service(source, vector_store)

        result = await service.ingest_paper("missing.00001")

        assert result.status == "error"
        assert "not found" in result.error
        assert result.chunk_count == 0

    async def test_pdf_parsing_error_returns_error(self, sample_paper, vector_store):
        source = FakePaperSource(
            sample_paper, chunks=[], extract_error=PDFParsingError("Corrupt PDF")
        )
        service = make_service(source, vector_store)

        result = await service.ingest_paper(sample_paper.arxiv_id)

        assert result.status == "error"
        assert result.error == "Corrupt PDF"

    async def test_unexpected_error_returns_generic_message(self, sample_paper, vector_store):
        """An unexpected error must not leak its raw text (e.g. a DB driver message)."""
        source = FakePaperSource(
            sample_paper,
            chunks=[],
            fetch_errors={
                sample_paper.arxiv_id: RuntimeError(
                    'duplicate key value violates unique constraint "papers_arxiv_id_key"'
                )
            },
        )
        service = make_service(source, vector_store)

        result = await service.ingest_paper(sample_paper.arxiv_id)

        assert result.status == "error"
        assert "internal error" in result.error.lower()
        assert "constraint" not in result.error, "raw driver detail must not reach the client"

    async def test_duplicate_in_progress_is_rejected(
        self, sample_paper, sample_chunks, vector_store
    ):
        gate = asyncio.Event()
        source = FakePaperSource(sample_paper, sample_chunks, fetch_gate=gate)
        service = make_service(source, vector_store)

        first = asyncio.create_task(service.ingest_paper(sample_paper.arxiv_id))
        await asyncio.sleep(0)  # let the first task block on the fetch gate

        # Same ID (with different casing/whitespace) must be rejected while in progress
        duplicate = await service.ingest_paper(f"  {sample_paper.arxiv_id.upper()}  ")
        assert duplicate.status == "error"
        assert "already being ingested" in duplicate.error

        gate.set()
        result = await first
        assert result.status == "success"

    async def test_in_progress_cleared_after_completion(
        self, sample_paper, sample_chunks, vector_store
    ):
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        first = await service.ingest_paper(sample_paper.arxiv_id)
        second = await service.ingest_paper(sample_paper.arxiv_id)

        assert first.status == "success"
        assert second.status == "success"

    async def test_in_progress_cleared_after_failure(
        self, sample_paper, sample_chunks, vector_store
    ):
        source = FakePaperSource(
            sample_paper,
            sample_chunks,
            fetch_errors={sample_paper.arxiv_id: PaperNotFoundError("not found")},
        )
        service = make_service(source, vector_store)

        failed = await service.ingest_paper(sample_paper.arxiv_id)
        assert failed.status == "error"

        # The ID must not be stuck in the in-progress set after a failure
        recovered = FakePaperSource(sample_paper, sample_chunks)
        service._paper_source = recovered  # noqa: SLF001 - swap source to prove retry works
        retried = await service.ingest_paper(sample_paper.arxiv_id)
        assert retried.status == "success"


class TestIngestPapers:
    """Tests for IngestionService.ingest_papers."""

    async def test_batch_partitions_successes_and_errors(
        self, sample_paper, sample_chunks, vector_store
    ):
        source = FakePaperSource(
            sample_paper,
            sample_chunks,
            fetch_errors={"bad.00001": PaperNotFoundError("bad.00001 not found")},
        )
        service = make_service(source, vector_store)

        result = await service.ingest_papers([sample_paper.arxiv_id, "bad.00001"])

        assert len(result.ingested) == 1
        assert result.ingested[0].status == "success"
        assert len(result.errors) == 1
        assert result.errors[0].arxiv_id == "bad.00001"

    async def test_empty_batch(self, sample_paper, sample_chunks, vector_store):
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        result = await service.ingest_papers([])

        assert result.ingested == []
        assert result.errors == []


class TestSearchAndIngest:
    """Tests for IngestionService.search_and_ingest."""

    async def test_search_results_are_ingested(self, sample_paper, sample_chunks, vector_store):
        source = FakePaperSource(sample_paper, sample_chunks)
        service = make_service(source, vector_store)

        result = await service.search_and_ingest("attention mechanisms", max_results=1)

        assert len(result.ingested) == 1
        assert result.ingested[0].arxiv_id == sample_paper.arxiv_id
        assert len(vector_store.added_chunks) == len(sample_chunks)

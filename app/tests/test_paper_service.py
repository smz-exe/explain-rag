"""Tests for PaperService, which owns the paper corpus rules.

The HTTP router used to hold these: 404-on-absence, the search preview length,
and turning a paper-source failure into something safe to hand a client.
"""

import pytest

from src.application.paper_service import (
    ABSTRACT_PREVIEW_CHARS,
    PaperSearchError,
    PaperSearchUnavailableError,
    PaperService,
)
from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort
from tests.conftest import MockVectorStorePort

LONG_ABSTRACT = "Retrieval quality drives generation quality. " * 20  # ~900 chars


def _paper(arxiv_id: str = "1706.03762", abstract: str = "An abstract.") -> Paper:
    """Build a paper with the fields a search hit carries."""
    return Paper(
        id=f"paper-{arxiv_id}",
        arxiv_id=arxiv_id,
        title="Attention Is All You Need",
        authors=["Vaswani, A."],
        abstract=abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


class StubPaperSource(PaperSourcePort):
    """Paper source returning fixed results, or failing on demand."""

    def __init__(self, papers: list[Paper] | None = None, error: Exception | None = None):
        self._papers = papers or [_paper()]
        self._error = error
        self.search_calls: list[tuple[str, int]] = []

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Not used by these tests."""
        raise PaperNotFoundError(arxiv_id)

    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """Record the call, then return the fixture or raise the fixture error."""
        self.search_calls.append((query, max_results))
        if self._error is not None:
            raise self._error
        return self._papers[:max_results]

    async def extract_chunks(self, paper, chunk_size, chunk_overlap):
        """Not used by these tests."""
        return []


class TestListPapers:
    """Listing the ingested corpus."""

    @pytest.mark.asyncio
    async def test_returns_stored_papers(self, sample_chunks):
        """The service hands back what the store holds."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=sample_chunks))

        papers = await service.list_papers()

        assert [p.paper_id for p in papers] == ["paper-001"]
        assert papers[0].chunk_count == 3

    @pytest.mark.asyncio
    async def test_lists_every_paper_with_its_own_count(self, mixed_paper_chunks):
        """Each paper reports its own chunk count, not the corpus total."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=mixed_paper_chunks))

        counts = {p.paper_id: p.chunk_count for p in await service.list_papers()}

        assert counts == {"paper-001": 3, "paper-002": 2}


class TestDeletePaper:
    """Deletion, and the distinction between 'absent' and 'had no chunks'."""

    @pytest.mark.asyncio
    async def test_returns_deleted_chunk_count(self, sample_chunks):
        """A successful delete reports how many chunks went with it."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=sample_chunks))

        assert await service.delete_paper("paper-001") == 3

    @pytest.mark.asyncio
    async def test_unknown_paper_raises(self, sample_chunks):
        """Absence is an error the caller must handle, not a count of zero."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=sample_chunks))

        with pytest.raises(PaperNotFoundError):
            await service.delete_paper("nonexistent-paper")

    @pytest.mark.asyncio
    async def test_second_delete_raises(self, sample_chunks):
        """Deleting twice: chunks removed, then the paper is gone."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=sample_chunks))

        assert await service.delete_paper("paper-001") == 3
        with pytest.raises(PaperNotFoundError):
            await service.delete_paper("paper-001")


class TestSearchPapers:
    """Searching the paper source for candidates to ingest."""

    @pytest.mark.asyncio
    async def test_returns_previews(self, sample_chunks):
        """Hits come back as previews carrying the fields the UI shows."""
        service = PaperService(
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            paper_source=StubPaperSource(),
        )

        previews = await service.search_papers("attention", 5)

        assert len(previews) == 1
        assert previews[0].arxiv_id == "1706.03762"
        assert previews[0].title == "Attention Is All You Need"
        assert previews[0].authors == ["Vaswani, A."]
        assert previews[0].url == "https://arxiv.org/abs/1706.03762"

    @pytest.mark.asyncio
    async def test_passes_max_results_through(self, sample_chunks):
        """The cap reaches the source rather than being applied after the fact."""
        source = StubPaperSource()
        service = PaperService(
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            paper_source=source,
        )

        await service.search_papers("attention", 3)

        assert source.search_calls == [("attention", 3)]

    @pytest.mark.asyncio
    async def test_long_abstract_is_truncated(self, sample_chunks):
        """A preview carries the gist, not the whole abstract."""
        service = PaperService(
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            paper_source=StubPaperSource(papers=[_paper(abstract=LONG_ABSTRACT)]),
        )

        abstract = (await service.search_papers("retrieval", 5))[0].abstract

        assert len(abstract) == ABSTRACT_PREVIEW_CHARS + 3
        assert abstract.endswith("...")
        assert abstract[:ABSTRACT_PREVIEW_CHARS] == LONG_ABSTRACT[:ABSTRACT_PREVIEW_CHARS]

    @pytest.mark.asyncio
    async def test_short_abstract_is_left_alone(self, sample_chunks):
        """No ellipsis is appended to an abstract that already fits."""
        service = PaperService(
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            paper_source=StubPaperSource(papers=[_paper(abstract="Short.")]),
        )

        assert (await service.search_papers("attention", 5))[0].abstract == "Short."

    @pytest.mark.asyncio
    async def test_source_failure_becomes_search_error(self, sample_chunks):
        """The source's own exception does not escape to the caller.

        Letting it through is how adapter internals (URLs, driver messages)
        reached clients before.
        """
        service = PaperService(
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            paper_source=StubPaperSource(error=RuntimeError("https://export.arxiv.org timed out")),
        )

        with pytest.raises(PaperSearchError) as exc_info:
            await service.search_papers("attention", 5)

        assert "export.arxiv.org" not in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_without_a_source_search_is_unavailable(self, sample_chunks):
        """No configured source is a distinct condition from a failed search."""
        service = PaperService(vector_store=MockVectorStorePort(chunks=sample_chunks))

        with pytest.raises(PaperSearchUnavailableError):
            await service.search_papers("attention", 5)

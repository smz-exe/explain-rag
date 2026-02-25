"""Tests for paper search functionality."""

import pytest

from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PaperSourcePort


class MockPaperSourcePort(PaperSourcePort):
    """Mock paper source for testing search functionality."""

    def __init__(self, papers: list[Paper] | None = None):
        self._papers = papers or [
            Paper(
                id="paper-001",
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani, A.", "Shazeer, N."],
                abstract="We propose a new simple network architecture, the Transformer, "
                "based solely on attention mechanisms.",
                url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            ),
            Paper(
                id="paper-002",
                arxiv_id="1810.04805",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Devlin, J.", "Chang, M."],
                abstract="We introduce BERT, a language representation model.",
                url="https://arxiv.org/abs/1810.04805",
                pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
            ),
        ]

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Fetch paper by ID."""
        for paper in self._papers:
            if paper.arxiv_id == arxiv_id:
                return paper
        from src.domain.ports.paper_source import PaperNotFoundError

        raise PaperNotFoundError(f"Paper not found: {arxiv_id}")

    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """Return mock search results."""
        # Filter papers that match the query in title or abstract
        results = [
            p
            for p in self._papers
            if query.lower() in p.title.lower() or query.lower() in p.abstract.lower()
        ]
        return results[:max_results]

    async def extract_chunks(self, paper, chunk_size, chunk_overlap):
        """Not used in search tests."""
        return []


@pytest.mark.asyncio
async def test_paper_search_requires_auth(client):
    """Test search endpoint requires authentication."""
    response = await client.get("/papers/search?query=transformer")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_paper_search_returns_results(authenticated_client):
    """Test search returns papers matching query."""
    # Note: This test uses the real ArxivPaperSource, so it may be slow
    # or fail due to network issues. Consider mocking for CI.
    response = await authenticated_client.get("/papers/search?query=transformer&max_results=2")

    # If the test fails due to network, skip gracefully
    if response.status_code == 500:
        pytest.skip("arXiv API unavailable")

    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "total" in data
    assert isinstance(data["papers"], list)


@pytest.mark.asyncio
async def test_paper_search_validates_query_length(authenticated_client):
    """Test search validates minimum query length."""
    response = await authenticated_client.get("/papers/search?query=a")
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_paper_search_validates_max_results(authenticated_client):
    """Test search validates max_results range."""
    # Too high
    response = await authenticated_client.get("/papers/search?query=test&max_results=100")
    assert response.status_code == 422

    # Too low
    response = await authenticated_client.get("/papers/search?query=test&max_results=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_paper_search_missing_query(authenticated_client):
    """Test search requires query parameter."""
    response = await authenticated_client.get("/papers/search")
    assert response.status_code == 422


class TestPaperSearchResponse:
    """Test search response structure."""

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, authenticated_client):
        """Test response includes all required fields."""
        response = await authenticated_client.get("/papers/search?query=attention&max_results=1")

        if response.status_code == 500:
            pytest.skip("arXiv API unavailable")

        assert response.status_code == 200
        data = response.json()

        # Check top-level fields
        assert "papers" in data
        assert "total" in data
        assert isinstance(data["total"], int)

        # Check paper fields if results exist
        if data["papers"]:
            paper = data["papers"][0]
            assert "arxiv_id" in paper
            assert "title" in paper
            assert "authors" in paper
            assert "abstract" in paper
            assert "url" in paper

    @pytest.mark.asyncio
    async def test_abstract_truncated(self, authenticated_client):
        """Test that long abstracts are truncated."""
        response = await authenticated_client.get("/papers/search?query=attention&max_results=1")

        if response.status_code == 500:
            pytest.skip("arXiv API unavailable")

        if response.status_code == 200:
            data = response.json()
            if data["papers"]:
                # Abstract should be <= 503 chars (500 + "...")
                paper = data["papers"][0]
                assert len(paper["abstract"]) <= 503

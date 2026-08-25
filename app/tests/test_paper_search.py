"""Tests for paper search functionality.

Uses a module-level `app` fixture that injects MockPaperSourcePort, so no
test here ever reaches the real arXiv API.
"""

import pytest

from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort
from src.main import create_app
from tests.conftest import (
    MockClusteringPort,
    MockCoordinatesStoragePort,
    MockDimensionalityReductionPort,
    MockEmbeddingPort,
    MockEvaluationPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockRerankerPort,
    MockVectorStorePort,
)

LONG_ABSTRACT = "Retrieval quality drives generation quality. " * 20  # ~900 chars


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
            Paper(
                id="paper-003",
                arxiv_id="2005.11401",
                title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                authors=["Lewis, P."],
                abstract=LONG_ABSTRACT,
                url="https://arxiv.org/abs/2005.11401",
                pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
            ),
        ]

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Fetch paper by ID."""
        for paper in self._papers:
            if paper.arxiv_id == arxiv_id:
                return paper
        raise PaperNotFoundError(f"Paper not found: {arxiv_id}")

    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """Return mock search results matching query in title or abstract."""
        results = [
            p
            for p in self._papers
            if query.lower() in p.title.lower() or query.lower() in p.abstract.lower()
        ]
        return results[:max_results]

    async def extract_chunks(self, paper, chunk_size, chunk_overlap):
        """Not used in search tests."""
        return []


@pytest.fixture
def app(sample_chunks):
    """Test app with a mock paper source (shadows the conftest app fixture)."""
    return create_app(
        embedding=MockEmbeddingPort(),
        vector_store=MockVectorStorePort(chunks=sample_chunks),
        paper_source=MockPaperSourcePort(),
        llm=MockLLMPort(),
        faithfulness=MockFaithfulnessPort(),
        reranker=MockRerankerPort(),
        evaluator=MockEvaluationPort(),
        query_storage=MockQueryStoragePort(),
        coordinates_storage=MockCoordinatesStoragePort(),
        dim_reducer=MockDimensionalityReductionPort(),
        clusterer=MockClusteringPort(),
    )


@pytest.mark.asyncio
async def test_paper_search_requires_auth(client):
    """Test search endpoint requires authentication."""
    response = await client.get("/papers/search?query=transformer")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_paper_search_returns_results(authenticated_client):
    """Test search returns papers matching query."""
    response = await authenticated_client.get("/papers/search?query=transformer&max_results=5")

    assert response.status_code == 200
    data = response.json()
    returned_ids = [p["arxiv_id"] for p in data["papers"]]
    assert returned_ids == ["1706.03762", "1810.04805"]
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_paper_search_respects_max_results(authenticated_client):
    """Test search caps results at max_results."""
    response = await authenticated_client.get("/papers/search?query=transformer&max_results=1")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["papers"][0]["arxiv_id"] == "1706.03762"


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

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        paper = data["papers"][0]
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["title"] == "Attention Is All You Need"
        assert paper["authors"] == ["Vaswani, A.", "Shazeer, N."]
        assert "abstract" in paper
        assert paper["url"] == "https://arxiv.org/abs/1706.03762"

    @pytest.mark.asyncio
    async def test_abstract_truncated(self, authenticated_client):
        """Test that abstracts longer than 500 chars are truncated with an ellipsis."""
        response = await authenticated_client.get("/papers/search?query=retrieval&max_results=5")

        assert response.status_code == 200
        papers = {p["arxiv_id"]: p for p in response.json()["papers"]}
        truncated = papers["2005.11401"]["abstract"]
        assert len(truncated) == 503  # 500 chars + "..."
        assert truncated.endswith("...")
        assert truncated[:500] == LONG_ABSTRACT[:500]

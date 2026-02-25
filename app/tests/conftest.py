"""Test fixtures for ExplainRAG tests."""

import os

# Set required environment variables before importing app
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only-min-32-chars")
os.environ.setdefault("ADMIN_USERNAME", "admin")
# bcrypt hash for "testpassword"
os.environ.setdefault(
    "ADMIN_PASSWORD_HASH",
    "$2b$12$u9la33VyaVebIX4B9Y6w1.J1YQd/UzQalxP4q17qqT8apDiKTCg3e",
)
# Test API key (not a real key, just for validation)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-testing-only")

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import ClaimVerification, FaithfulnessResult
from src.domain.entities.paper import Paper
from src.domain.entities.query import Citation, GenerationResult, QueryResponse
from src.domain.entities.coordinates import Cluster, PaperCoordinates
from src.domain.ports.clustering import ClusteringPort
from src.domain.ports.coordinates_storage import CoordinatesStoragePort
from src.domain.ports.dimensionality_reduction import DimensionalityReductionPort
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.faithfulness import FaithfulnessPort
from src.domain.ports.llm import LLMPort
from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.reranker import RerankerPort
from src.domain.ports.vector_store import VectorStorePort
from src.main import create_app


@pytest.fixture
def app():
    """Create a test application instance."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
async def authenticated_client(app):
    """Create an authenticated async test client with admin credentials."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Login to get the auth cookie
        response = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "testpassword"},
        )
        assert response.status_code == 200, f"Login failed: {response.json()}"
        yield client


# Sample data fixtures


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        id="paper-001",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
        abstract="We propose a new simple network architecture...",
        url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk-001",
            paper_id="paper-001",
            content="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            chunk_index=0,
            section="Introduction",
            metadata={"paper_title": "Attention Is All You Need"},
        ),
        Chunk(
            id="chunk-002",
            paper_id="paper-001",
            content="Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence.",
            chunk_index=1,
            section="Background",
            metadata={"paper_title": "Attention Is All You Need"},
        ),
        Chunk(
            id="chunk-003",
            paper_id="paper-001",
            content="The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers.",
            chunk_index=2,
            section="Model Architecture",
            metadata={"paper_title": "Attention Is All You Need"},
        ),
    ]


# Mock adapters


class MockEmbeddingPort(EmbeddingPort):
    """Mock embedding adapter for testing."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embed_calls: list[str] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings based on text content."""
        self.embed_calls.extend(texts)
        return [[0.1 * (i + 1)] * self.embedding_dim for i in range(len(texts))]

    async def embed_query(self, query: str) -> list[float]:
        """Return a deterministic query embedding."""
        self.embed_calls.append(query)
        return [0.5] * self.embedding_dim


class MockVectorStorePort(VectorStorePort):
    """Mock vector store adapter for testing."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or []
        self.added_chunks: list[Chunk] = []

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Store chunks."""
        self.added_chunks.extend(chunks)
        self.chunks.extend(chunks)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return stored chunks with mock scores."""
        results = []
        for i, chunk in enumerate(self.chunks[:top_k]):
            # Apply filter if specified
            if filter and "paper_id" in filter:
                filter_values = filter["paper_id"].get("$in", [])
                if chunk.paper_id not in filter_values:
                    continue
            score = 0.9 - (i * 0.1)  # Decreasing scores
            results.append((chunk, score))
        return results

    async def get_stats(self) -> dict:
        """Return mock stats."""
        return {"chunk_count": len(self.chunks), "paper_count": 1}

    async def list_papers(self) -> list[dict]:
        """Return mock paper list."""
        return [{"paper_id": "paper-001", "title": "Test Paper", "chunk_count": 3}]

    async def delete_paper(self, paper_id: str) -> int:
        """Delete chunks for a paper."""
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.paper_id != paper_id]
        return original_count - len(self.chunks)

    async def get_paper_embeddings(self) -> list[tuple[str, list[float]]]:
        """Return mock paper embeddings."""
        # Group by paper_id and return mock mean embeddings
        paper_ids = set(c.paper_id for c in self.chunks)
        return [(pid, [0.5] * 384) for pid in paper_ids]


class MockDimensionalityReductionPort(DimensionalityReductionPort):
    """Mock dimensionality reduction adapter for testing."""

    def __init__(self):
        self._fitted = False
        self._fit_transform_calls: list[list[list[float]]] = []
        self._transform_calls: list[list[list[float]]] = []

    def is_fitted(self) -> bool:
        """Check if fitted."""
        return self._fitted

    async def fit_transform(
        self,
        embeddings: list[list[float]],
        n_components: int = 3,
    ) -> list[tuple[float, float, float]]:
        """Return mock 3D coordinates."""
        self._fit_transform_calls.append(embeddings)
        self._fitted = True
        # Return deterministic coordinates based on index
        return [(float(i), float(i) * 0.5, float(i) * -0.5) for i in range(len(embeddings))]

    async def transform(
        self,
        embeddings: list[list[float]],
    ) -> list[tuple[float, float, float]]:
        """Transform new embeddings."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        self._transform_calls.append(embeddings)
        return [(0.0, 0.0, 0.0) for _ in embeddings]


class MockClusteringPort(ClusteringPort):
    """Mock clustering adapter for testing."""

    def __init__(self, cluster_labels: list[int] | None = None):
        self._labels = cluster_labels
        self._cluster_calls: list[list[list[float]]] = []
        self._clustered = False

    async def cluster(
        self,
        embeddings: list[list[float]],
    ) -> list[int]:
        """Return mock cluster labels."""
        self._cluster_calls.append(embeddings)
        self._clustered = True
        if self._labels is not None:
            return self._labels[: len(embeddings)]
        # Default: assign to clusters 0, 1, 0, 1, ...
        return [i % 2 for i in range(len(embeddings))]

    async def get_cluster_count(self) -> int:
        """Get cluster count."""
        if not self._clustered:
            raise RuntimeError("Not clustered")
        if self._labels is not None:
            unique = set(self._labels)
            unique.discard(-1)
            return len(unique)
        return 2  # Default mock has 2 clusters


class MockLLMPort(LLMPort):
    """Mock LLM adapter for testing."""

    def __init__(self, answer: str | None = None, citations: list[Citation] | None = None):
        self.answer = answer or "Self-attention is a mechanism [1]. It relates positions [2]."
        self.citations = citations or [
            Citation(
                claim="Self-attention is a mechanism", chunk_ids=["chunk-001"], confidence=0.9
            ),
            Citation(claim="It relates positions", chunk_ids=["chunk-002"], confidence=0.85),
        ]
        self.generate_calls: list[tuple[str, list[Chunk]]] = []

    async def generate(self, question: str, chunks: list[Chunk]) -> GenerationResult:
        """Return mock generation result."""
        self.generate_calls.append((question, chunks))
        return GenerationResult(
            answer=self.answer,
            citations=self.citations,
        )


class MockFaithfulnessPort(FaithfulnessPort):
    """Mock faithfulness adapter for testing."""

    def __init__(self, score: float = 0.9, claims: list[ClaimVerification] | None = None):
        self.score = score
        self.claims = claims or [
            ClaimVerification(
                claim="Self-attention is a mechanism",
                verdict="supported",
                evidence_chunk_ids=["chunk-001"],
                reasoning="Directly stated in chunk",
            ),
            ClaimVerification(
                claim="It relates positions",
                verdict="supported",
                evidence_chunk_ids=["chunk-002"],
                reasoning="Matches chunk content",
            ),
        ]
        self.verify_calls: list[tuple[str, list[Chunk]]] = []

    async def verify(self, answer: str, chunks: list[Chunk]) -> FaithfulnessResult:
        """Return mock faithfulness result."""
        self.verify_calls.append((answer, chunks))
        return FaithfulnessResult(score=self.score, claims=self.claims)


class MockRerankerPort(RerankerPort):
    """Mock reranker adapter for testing."""

    def __init__(self, reverse_order: bool = True):
        """Initialize mock reranker.

        Args:
            reverse_order: If True, reverses the order of chunks to simulate reranking.
        """
        self.reverse_order = reverse_order
        self.rerank_calls: list[tuple[str, list[Chunk]]] = []

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return reranked chunks with mock scores."""
        self.rerank_calls.append((query, chunks))

        reranked = list(reversed(chunks)) if self.reverse_order else chunks

        # Assign decreasing scores
        results = [(chunk, 0.95 - (i * 0.05)) for i, chunk in enumerate(reranked)]

        if top_k is not None:
            results = results[:top_k]

        return results


class MockQueryStoragePort(QueryStoragePort):
    """Mock query storage adapter for testing."""

    def __init__(self):
        self.queries: dict[str, QueryResponse] = {}
        self.store_calls: list[QueryResponse] = []

    async def store(self, response: QueryResponse) -> None:
        """Store a query response."""
        self.store_calls.append(response)
        self.queries[response.query_id] = response

    async def get(self, query_id: str) -> QueryResponse | None:
        """Retrieve a query response by ID."""
        return self.queries.get(query_id)

    async def list_recent(self, limit: int = 20) -> list[dict]:
        """List recent queries."""
        return [
            {
                "query_id": q.query_id,
                "question": q.question,
                "answer_preview": q.answer[:200] if q.answer else "",
                "created_at": "2025-01-01T00:00:00Z",
            }
            for q in list(self.queries.values())[-limit:]
        ]

    async def delete(self, query_id: str) -> bool:
        """Delete a query."""
        if query_id in self.queries:
            del self.queries[query_id]
            return True
        return False

    async def count(self) -> int:
        """Get total query count."""
        return len(self.queries)


class MockCoordinatesStoragePort(CoordinatesStoragePort):
    """Mock coordinates storage adapter for testing."""

    def __init__(
        self,
        initial_coordinates: list[PaperCoordinates] | None = None,
        initial_clusters: list[Cluster] | None = None,
        initial_computed_at: "datetime | None" = None,
    ):
        from datetime import datetime

        self.coordinates: list[PaperCoordinates] = initial_coordinates or []
        self.clusters: list[Cluster] = initial_clusters or []
        self.computed_at: datetime | None = initial_computed_at
        self.load_calls: int = 0
        self.save_calls: list[tuple[list[PaperCoordinates], list[Cluster], datetime]] = []
        self.clear_calls: int = 0

    async def load(
        self,
    ) -> tuple[list[PaperCoordinates], list[Cluster], "datetime | None"]:
        """Load stored coordinates and clusters."""
        self.load_calls += 1
        return self.coordinates.copy(), self.clusters.copy(), self.computed_at

    async def save(
        self,
        coordinates: list[PaperCoordinates],
        clusters: list[Cluster],
        computed_at: "datetime",
    ) -> None:
        """Save coordinates and clusters."""
        self.save_calls.append((coordinates, clusters, computed_at))
        self.coordinates = coordinates.copy()
        self.clusters = clusters.copy()
        self.computed_at = computed_at

    async def clear(self) -> None:
        """Clear all stored coordinates and clusters."""
        self.clear_calls += 1
        self.coordinates = []
        self.clusters = []
        self.computed_at = None


# Fixtures for mock adapters


@pytest.fixture
def mock_embedding() -> MockEmbeddingPort:
    """Create a mock embedding adapter."""
    return MockEmbeddingPort()


@pytest.fixture
def mock_vector_store(sample_chunks) -> MockVectorStorePort:
    """Create a mock vector store with sample chunks."""
    return MockVectorStorePort(chunks=sample_chunks)


@pytest.fixture
def mock_llm() -> MockLLMPort:
    """Create a mock LLM adapter."""
    return MockLLMPort()


@pytest.fixture
def mock_faithfulness() -> MockFaithfulnessPort:
    """Create a mock faithfulness adapter."""
    return MockFaithfulnessPort()


@pytest.fixture
def mock_reranker() -> MockRerankerPort:
    """Create a mock reranker adapter."""
    return MockRerankerPort()


@pytest.fixture
def mock_query_storage() -> MockQueryStoragePort:
    """Create a mock query storage adapter."""
    return MockQueryStoragePort()


@pytest.fixture
def mock_dim_reduction() -> MockDimensionalityReductionPort:
    """Create a mock dimensionality reduction adapter."""
    return MockDimensionalityReductionPort()


@pytest.fixture
def mock_clustering() -> MockClusteringPort:
    """Create a mock clustering adapter."""
    return MockClusteringPort()


@pytest.fixture
def mock_coordinates_storage() -> MockCoordinatesStoragePort:
    """Create a mock coordinates storage adapter."""
    return MockCoordinatesStoragePort()

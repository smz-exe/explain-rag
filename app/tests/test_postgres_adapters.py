"""Tests for PostgreSQL adapters (vector store and query storage)."""

import os
import uuid

import pytest

from src.adapters.outbound.postgres_query_storage import PostgresQueryStorage
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult
from src.domain.entities.query import Citation, QueryResponse, RetrievedChunk

# Skip tests if DATABASE_URL is not set (e.g., in CI without Supabase)
DATABASE_URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL or "127.0.0.1:54322" not in DATABASE_URL,
    reason="Requires local Supabase database (DATABASE_URL not set or not local)",
)


class TestPostgresVectorStore:
    """Tests for PostgresVectorStore adapter."""

    @pytest.fixture
    async def vector_store(self):
        """Create a PostgresVectorStore instance for testing."""
        store = PostgresVectorStore(DATABASE_URL)
        yield store
        await store.close()

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        paper_id = str(uuid.uuid4())
        # Use unique arxiv_id for each test to avoid conflicts
        arxiv_id = f"test.{uuid.uuid4().hex[:8]}"
        return [
            Chunk(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                content=f"This is chunk {i} content about machine learning.",
                chunk_index=i,
                section=f"Section {i}",
                metadata={
                    "paper_title": "Test Paper on ML",
                    "arxiv_id": arxiv_id,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "authors": ["Author One", "Author Two"],
                    "abstract": "This is a test abstract.",
                },
            )
            for i in range(3)
        ]

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Create sample embeddings (384-dimensional for all-MiniLM-L6-v2)."""
        return [[0.1 * (i + 1)] * 384 for i in range(3)]

    async def test_get_stats_empty(self, vector_store: PostgresVectorStore):
        """Test get_stats on empty database."""
        stats = await vector_store.get_stats()
        assert "chunk_count" in stats
        assert "paper_count" in stats
        assert isinstance(stats["chunk_count"], int)
        assert isinstance(stats["paper_count"], int)

    async def test_list_papers_empty(self, vector_store: PostgresVectorStore):
        """Test list_papers on empty database."""
        papers = await vector_store.list_papers()
        assert isinstance(papers, list)

    async def test_add_and_search_chunks(
        self,
        vector_store: PostgresVectorStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test adding chunks and searching for them."""
        # Add chunks
        await vector_store.add_chunks(sample_chunks, sample_embeddings)

        # Search with similar embedding
        query_embedding = [0.15] * 384
        results = await vector_store.search(query_embedding, top_k=5)

        assert len(results) > 0
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

        # Cleanup
        await vector_store.delete_paper(sample_chunks[0].paper_id)

    async def test_delete_paper(
        self,
        vector_store: PostgresVectorStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test deleting a paper and its chunks."""
        # Add chunks
        await vector_store.add_chunks(sample_chunks, sample_embeddings)
        paper_id = sample_chunks[0].paper_id

        # Verify paper exists
        papers = await vector_store.list_papers()
        paper_ids = [p["paper_id"] for p in papers]
        assert paper_id in paper_ids

        # Delete paper
        deleted_count = await vector_store.delete_paper(paper_id)
        assert deleted_count == 3

        # Verify paper is gone
        papers = await vector_store.list_papers()
        paper_ids = [p["paper_id"] for p in papers]
        assert paper_id not in paper_ids

    async def test_search_with_paper_filter(
        self,
        vector_store: PostgresVectorStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test searching with paper_id filter."""
        # Add chunks
        await vector_store.add_chunks(sample_chunks, sample_embeddings)
        paper_id = sample_chunks[0].paper_id

        # Search with filter
        query_embedding = [0.15] * 384
        results = await vector_store.search(
            query_embedding, top_k=5, filter={"paper_id": paper_id}
        )

        assert len(results) > 0
        for chunk, _ in results:
            assert chunk.paper_id == paper_id

        # Cleanup
        await vector_store.delete_paper(paper_id)

    async def test_get_paper_embeddings(
        self,
        vector_store: PostgresVectorStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test getting mean embeddings per paper."""
        # Add chunks
        await vector_store.add_chunks(sample_chunks, sample_embeddings)
        paper_id = sample_chunks[0].paper_id

        # Get embeddings
        paper_embeddings = await vector_store.get_paper_embeddings()

        assert len(paper_embeddings) > 0
        found = False
        for pid, embedding in paper_embeddings:
            if pid == paper_id:
                found = True
                assert len(embedding) == 384
                break
        assert found

        # Cleanup
        await vector_store.delete_paper(paper_id)


class TestPostgresQueryStorage:
    """Tests for PostgresQueryStorage adapter."""

    @pytest.fixture
    async def query_storage(self):
        """Create a PostgresQueryStorage instance for testing."""
        storage = PostgresQueryStorage(DATABASE_URL)
        yield storage
        await storage.close()

    @pytest.fixture
    def sample_query_response(self) -> QueryResponse:
        """Create a sample QueryResponse for testing."""
        return QueryResponse(
            query_id=str(uuid.uuid4()),
            question="What is machine learning?",
            answer="Machine learning is a type of AI.",
            citations=[
                Citation(
                    claim="Machine learning is a type of AI.",
                    chunk_ids=["chunk-1"],
                    confidence=0.95,
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    paper_title="Test Paper",
                    content="Machine learning is a field of AI.",
                    similarity_score=0.9,
                    rerank_score=None,
                    original_rank=1,
                    rank=1,
                )
            ],
            faithfulness=FaithfulnessResult(score=0.95, claims=[]),
            trace=ExplanationTrace(
                embedding_time_ms=10.0,
                retrieval_time_ms=20.0,
                reranking_time_ms=None,
                generation_time_ms=100.0,
                faithfulness_time_ms=50.0,
                total_time_ms=180.0,
            ),
        )

    async def test_count_empty(self, query_storage: PostgresQueryStorage):
        """Test count on empty database."""
        count = await query_storage.count()
        assert isinstance(count, int)

    async def test_list_recent_empty(self, query_storage: PostgresQueryStorage):
        """Test list_recent on empty database."""
        queries = await query_storage.list_recent()
        assert isinstance(queries, list)

    async def test_store_and_get(
        self,
        query_storage: PostgresQueryStorage,
        sample_query_response: QueryResponse,
    ):
        """Test storing and retrieving a query."""
        # Store
        await query_storage.store(sample_query_response)

        # Get
        retrieved = await query_storage.get(sample_query_response.query_id)
        assert retrieved is not None
        assert retrieved.query_id == sample_query_response.query_id
        assert retrieved.question == sample_query_response.question
        assert retrieved.answer == sample_query_response.answer

        # Cleanup
        await query_storage.delete(sample_query_response.query_id)

    async def test_list_recent(
        self,
        query_storage: PostgresQueryStorage,
        sample_query_response: QueryResponse,
    ):
        """Test listing recent queries."""
        # Store
        await query_storage.store(sample_query_response)

        # List
        queries = await query_storage.list_recent(limit=10)
        assert len(queries) > 0

        found = False
        for q in queries:
            if q["query_id"] == sample_query_response.query_id:
                found = True
                assert q["question"] == sample_query_response.question
                break
        assert found

        # Cleanup
        await query_storage.delete(sample_query_response.query_id)

    async def test_delete(
        self,
        query_storage: PostgresQueryStorage,
        sample_query_response: QueryResponse,
    ):
        """Test deleting a query."""
        # Store
        await query_storage.store(sample_query_response)

        # Verify exists
        retrieved = await query_storage.get(sample_query_response.query_id)
        assert retrieved is not None

        # Delete
        deleted = await query_storage.delete(sample_query_response.query_id)
        assert deleted is True

        # Verify gone
        retrieved = await query_storage.get(sample_query_response.query_id)
        assert retrieved is None

    async def test_get_nonexistent(self, query_storage: PostgresQueryStorage):
        """Test getting a nonexistent query."""
        retrieved = await query_storage.get(str(uuid.uuid4()))
        assert retrieved is None

    async def test_delete_nonexistent(self, query_storage: PostgresQueryStorage):
        """Test deleting a nonexistent query."""
        deleted = await query_storage.delete(str(uuid.uuid4()))
        assert deleted is False

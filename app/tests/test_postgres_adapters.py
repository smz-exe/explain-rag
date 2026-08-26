"""Tests for PostgreSQL adapters (vector store and query storage)."""

import os
import uuid

import asyncpg
import pytest

from src.adapters.outbound.postgres_query_storage import PostgresQueryStorage
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.application.query_service import QueryService
from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult
from src.domain.entities.paper import Paper
from src.domain.entities.query import Citation, QueryRequest, QueryResponse, RetrievedChunk

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

    @pytest.fixture
    def sample_paper(self, sample_chunks: list[Chunk]) -> Paper:
        """The paper the sample chunks belong to."""
        arxiv_id = sample_chunks[0].metadata["arxiv_id"]
        return Paper(
            id=sample_chunks[0].paper_id,
            arxiv_id=arxiv_id,
            title="Test Paper on ML",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract.",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )

    @pytest.fixture
    def sample_paper_and_chunks(
        self, sample_paper: Paper, sample_chunks: list[Chunk]
    ) -> tuple[Paper, list[Chunk]]:
        return sample_paper, sample_chunks

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
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test adding chunks and searching for them."""
        # Add chunks
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)

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

    async def test_paper_metadata_is_persisted(
        self,
        vector_store: PostgresVectorStore,
        sample_paper_and_chunks,
        sample_embeddings: list[list[float]],
    ):
        """Authors, abstract, and links must survive ingestion.

        These used to be read out of the first chunk's free-form metadata dict,
        which the ingestion service never populated with them, so every stored
        paper had empty authors, abstract, url, and pdf_url in production while
        the adapter's own tests passed by setting those keys by hand.
        """
        paper, chunks = sample_paper_and_chunks
        await vector_store.add_chunks(paper, chunks, sample_embeddings)

        try:
            stored = [
                p for p in await vector_store.list_papers() if p["arxiv_id"] == paper.arxiv_id
            ]
            assert len(stored) == 1
            assert stored[0]["title"] == paper.title
            assert stored[0]["authors"] == paper.authors
            assert stored[0]["abstract"] == paper.abstract
            assert stored[0]["url"] == paper.url
            assert stored[0]["pdf_url"] == paper.pdf_url
        finally:
            await vector_store.delete_paper(paper.id)

    async def test_reingesting_same_arxiv_id_does_not_crash(
        self,
        vector_store: PostgresVectorStore,
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Re-ingesting an already-stored paper must not raise.

        The paper source mints a fresh Paper.id on every fetch, so a re-ingest
        arrives with a new paper_id but the same arxiv_id. The old code keyed
        existence on the new id and inserted with ON CONFLICT (id), so the
        arxiv_id UNIQUE constraint fired and asyncpg raised UniqueViolationError.
        """
        arxiv_id = sample_chunks[0].metadata["arxiv_id"]
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)

        # Simulate a second fetch of the same paper: new paper_id + chunk ids,
        # same arxiv_id, and updated content.
        second_paper_id = str(uuid.uuid4())
        refetched = [
            Chunk(
                id=str(uuid.uuid4()),
                paper_id=second_paper_id,
                content=f"Updated chunk {i} content.",
                chunk_index=i,
                section=f"Section {i}",
                metadata={"paper_title": "Test Paper on ML", "arxiv_id": arxiv_id},
            )
            for i in range(3)
        ]
        refetched_paper = sample_paper.model_copy(update={"id": second_paper_id})
        await vector_store.add_chunks(refetched_paper, refetched, sample_embeddings)

        try:
            papers = [p for p in await vector_store.list_papers() if p["arxiv_id"] == arxiv_id]
            assert len(papers) == 1, "re-ingestion must not create a duplicate paper row"

            canonical_id = papers[0]["paper_id"]
            results = await vector_store.search([0.15] * 384, top_k=10, paper_ids=[canonical_id])
            assert results, "chunks must be attached to the canonical paper id"
            assert all("Updated chunk" in chunk.content for chunk, _ in results)
        finally:
            await vector_store.delete_paper(sample_chunks[0].paper_id)
            await vector_store.delete_paper(second_paper_id)

    async def test_reingesting_fewer_chunks_leaves_no_stale_rows(
        self,
        vector_store: PostgresVectorStore,
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """A paper re-chunked into fewer pieces must not keep its old tail chunks."""
        arxiv_id = sample_chunks[0].metadata["arxiv_id"]
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)

        shrunk = [
            Chunk(
                id=str(uuid.uuid4()),
                paper_id=str(uuid.uuid4()),
                content="Only chunk now.",
                chunk_index=0,
                section="Section 0",
                metadata={"paper_title": "Test Paper on ML", "arxiv_id": arxiv_id},
            )
        ]
        shrunk_paper = sample_paper.model_copy(update={"id": shrunk[0].paper_id})
        await vector_store.add_chunks(shrunk_paper, shrunk, sample_embeddings[:1])

        try:
            papers = [p for p in await vector_store.list_papers() if p["arxiv_id"] == arxiv_id]
            assert papers[0]["chunk_count"] == 1
        finally:
            # The paper keeps its original id across re-ingestion, so clean up by that.
            await vector_store.delete_paper(sample_chunks[0].paper_id)

    async def test_failed_chunk_write_leaves_no_orphan_paper(
        self,
        vector_store: PostgresVectorStore,
        sample_paper: Paper,
        sample_chunks: list[Chunk],
    ):
        """A failure during chunk insert must roll back the paper row too.

        Without a transaction the papers INSERT committed on its own, leaving an
        orphan row that then blocked re-ingestion of that arxiv_id forever.
        """
        arxiv_id = sample_chunks[0].metadata["arxiv_id"]
        # 5 dimensions instead of 384: the embedding column rejects this.
        bad_embeddings = [[0.1] * 5 for _ in sample_chunks]

        with pytest.raises(asyncpg.PostgresError):
            await vector_store.add_chunks(sample_paper, sample_chunks, bad_embeddings)

        papers = [p for p in await vector_store.list_papers() if p["arxiv_id"] == arxiv_id]
        assert papers == [], "paper row must not survive a failed chunk write"

    async def test_malformed_paper_id_is_not_found_rather_than_an_error(
        self, vector_store: PostgresVectorStore
    ):
        """IDs arrive as path strings; a non-uuid cannot exist, it is not a crash.

        Binding one to a uuid column raises asyncpg.DataError, which surfaced
        as a 500 instead of a 404.
        """
        assert await vector_store.delete_paper("not-a-uuid") is None
        assert await vector_store.search([0.1] * 384, top_k=5, paper_ids=["not-a-uuid"]) == []

    async def test_deleting_a_paper_with_no_chunks_reports_it_was_deleted(
        self, vector_store: PostgresVectorStore, sample_paper: Paper, sample_chunks: list[Chunk]
    ):
        """Chunk count is not an existence check.

        delete_paper returned the pre-delete chunk count, so a paper with zero
        chunks was actually deleted while the API reported 404.
        """
        await vector_store.add_chunks(sample_paper, sample_chunks[:1], [[0.1] * 384])
        # Remove its chunks but leave the paper row in place.
        pool = await vector_store._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE paper_id = $1", sample_paper.id)

        deleted = await vector_store.delete_paper(sample_paper.id)

        assert deleted == 0, "the paper existed and had no chunks"
        assert await vector_store.delete_paper(sample_paper.id) is None, "now it is gone"

    async def test_delete_paper(
        self,
        vector_store: PostgresVectorStore,
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test deleting a paper and its chunks."""
        # Add chunks
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)
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
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test searching with paper_id filter."""
        # Add chunks
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)
        paper_id = sample_chunks[0].paper_id

        # Search with filter
        query_embedding = [0.15] * 384
        results = await vector_store.search(query_embedding, top_k=5, paper_ids=[paper_id])

        assert len(results) > 0
        for chunk, _ in results:
            assert chunk.paper_id == paper_id

        # Cleanup
        await vector_store.delete_paper(paper_id)

    async def test_get_paper_embeddings(
        self,
        vector_store: PostgresVectorStore,
        sample_paper: Paper,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ):
        """Test getting mean embeddings per paper."""
        # Add chunks
        await vector_store.add_chunks(sample_paper, sample_chunks, sample_embeddings)
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


class TestQueryServiceAgainstPostgres:
    """QueryService composed with the real vector store adapter.

    Every other service-level test substitutes MockVectorStorePort, so a
    contract mismatch between the service and PostgresVectorStore stays
    invisible. These tests exercise the real binding path.
    """

    @pytest.fixture
    async def vector_store(self):
        store = PostgresVectorStore(DATABASE_URL)
        yield store
        await store.close()

    @pytest.fixture
    def two_papers(self) -> tuple[list[Chunk], list[list[float]]]:
        """Two distinct papers with their chunks, so scoping is observable."""
        papers: list[Paper] = []
        chunks: list[Chunk] = []
        for paper_index in range(2):
            paper_id = str(uuid.uuid4())
            arxiv_id = f"test.{uuid.uuid4().hex[:8]}"
            papers.append(
                Paper(
                    id=paper_id,
                    arxiv_id=arxiv_id,
                    title=f"Test Paper {paper_index}",
                    authors=["Author One"],
                    abstract="Abstract.",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                )
            )
            chunks.extend(
                Chunk(
                    id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    content=f"Paper {paper_index} chunk {i} about machine learning.",
                    chunk_index=i,
                    section=f"Section {i}",
                    metadata={"paper_title": f"Test Paper {paper_index}", "arxiv_id": arxiv_id},
                )
                for i in range(2)
            )
        embeddings = [[0.1 * (i + 1)] * 384 for i in range(len(chunks))]
        return papers, chunks, embeddings

    async def _make_service(self, vector_store: PostgresVectorStore) -> QueryService:
        from tests.conftest import MockEmbeddingPort, MockFaithfulnessPort, MockLLMPort

        return QueryService(
            embedding=MockEmbeddingPort(),
            vector_store=vector_store,
            llm=MockLLMPort(),
            faithfulness=MockFaithfulnessPort(),
        )

    async def test_query_scoped_to_paper_ids_hits_the_real_adapter(
        self,
        vector_store: PostgresVectorStore,
        two_papers: tuple[list[Paper], list[Chunk], list[list[float]]],
    ):
        """A paper-scoped query must reach Postgres and exclude other papers.

        Regression test: the service used to send {"paper_id": {"$in": [...]}},
        which asyncpg cannot bind to $2::uuid[], so every scoped query raised
        DataError and surfaced as HTTP 500.
        """
        papers, chunks, embeddings = two_papers
        wanted_paper = chunks[0].paper_id
        other_paper = chunks[-1].paper_id
        assert wanted_paper != other_paper

        await vector_store.add_chunks(papers[0], chunks[:2], embeddings[:2])
        await vector_store.add_chunks(papers[1], chunks[2:], embeddings[2:])
        try:
            service = await self._make_service(vector_store)
            response = await service.query(
                QueryRequest(question="What is machine learning?", paper_ids=[wanted_paper])
            )

            returned_papers = {rc.paper_id for rc in response.retrieved_chunks}
            assert returned_papers == {wanted_paper}
        finally:
            await vector_store.delete_paper(wanted_paper)
            await vector_store.delete_paper(other_paper)

    async def test_query_without_scope_reaches_multiple_papers(
        self,
        vector_store: PostgresVectorStore,
        two_papers: tuple[list[Paper], list[Chunk], list[list[float]]],
    ):
        """An unscoped query must not be restricted to a single paper."""
        papers, chunks, embeddings = two_papers
        first_paper = chunks[0].paper_id
        second_paper = chunks[-1].paper_id

        await vector_store.add_chunks(papers[0], chunks[:2], embeddings[:2])
        await vector_store.add_chunks(papers[1], chunks[2:], embeddings[2:])
        try:
            service = await self._make_service(vector_store)
            response = await service.query(
                QueryRequest(question="What is machine learning?", top_k=50)
            )

            returned_papers = {rc.paper_id for rc in response.retrieved_chunks}
            assert {first_paper, second_paper} <= returned_papers
        finally:
            await vector_store.delete_paper(first_paper)
            await vector_store.delete_paper(second_paper)


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

    async def test_list_by_verification_status_separates_lifecycle_states(
        self,
        query_storage: PostgresQueryStorage,
        sample_query_response: QueryResponse,
    ):
        """Startup reconciliation depends on this JSONB query being exactly right.

        Completed results store the bare FaithfulnessResult (with a "score"
        key); pending and failed store a status envelope instead.
        """
        completed = sample_query_response
        pending = sample_query_response.model_copy(
            update={
                "query_id": str(uuid.uuid4()),
                "faithfulness": None,
                "faithfulness_status": "pending",
            }
        )
        failed = sample_query_response.model_copy(
            update={
                "query_id": str(uuid.uuid4()),
                "faithfulness": None,
                "faithfulness_status": "failed",
            }
        )
        for response in (completed, pending, failed):
            await query_storage.store(response)

        try:
            pending_ids = await query_storage.list_by_verification_status("pending")
            failed_ids = await query_storage.list_by_verification_status("failed")
            completed_ids = await query_storage.list_by_verification_status("completed")

            assert pending.query_id in pending_ids
            assert pending.query_id not in failed_ids
            assert pending.query_id not in completed_ids

            assert failed.query_id in failed_ids
            assert failed.query_id not in pending_ids

            assert completed.query_id in completed_ids
            assert completed.query_id not in pending_ids
        finally:
            for response in (completed, pending, failed):
                await query_storage.delete(response.query_id)

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

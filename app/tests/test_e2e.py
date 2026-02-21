"""End-to-end tests for the query pipeline."""

import pytest

from src.application.query_service import QueryService
from src.domain.entities.query import QueryRequest
from tests.conftest import (
    MockEmbeddingPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockRerankerPort,
    MockVectorStorePort,
)


@pytest.fixture
def query_service(sample_chunks) -> QueryService:
    """Create a QueryService with all mock adapters."""
    embedding = MockEmbeddingPort()
    vector_store = MockVectorStorePort(chunks=sample_chunks)
    llm = MockLLMPort()
    faithfulness = MockFaithfulnessPort()
    reranker = MockRerankerPort()
    query_storage = MockQueryStoragePort()

    return QueryService(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        faithfulness=faithfulness,
        reranker=reranker,
        query_storage=query_storage,
    )


class TestFullQueryPipeline:
    """Test the full query pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self, query_service):
        """Test complete query pipeline returns expected response."""
        request = QueryRequest(question="What is self-attention?")
        response = await query_service.query(request)

        # Basic response structure
        assert response.query_id is not None
        assert response.question == "What is self-attention?"
        assert response.answer is not None

        # Citations
        assert response.citations is not None
        assert len(response.citations) > 0

        # Retrieved chunks
        assert response.retrieved_chunks is not None
        assert len(response.retrieved_chunks) > 0

        # Faithfulness
        assert response.faithfulness is not None
        assert 0.0 <= response.faithfulness.score <= 1.0

        # Trace
        assert response.trace is not None
        assert response.trace.embedding_time_ms >= 0
        assert response.trace.retrieval_time_ms >= 0
        assert response.trace.generation_time_ms >= 0
        assert response.trace.faithfulness_time_ms >= 0
        assert response.trace.total_time_ms >= 0

    @pytest.mark.asyncio
    async def test_query_with_reranking_enabled(self, query_service):
        """Test query pipeline with reranking enabled."""
        request = QueryRequest(
            question="What is self-attention?",
            enable_reranking=True,
        )
        response = await query_service.query(request)

        # Reranking should be reflected in trace
        assert response.trace.reranking_time_ms is not None
        assert response.trace.reranking_time_ms >= 0

        # Chunks should have rerank scores
        for chunk in response.retrieved_chunks:
            assert chunk.rerank_score is not None

    @pytest.mark.asyncio
    async def test_query_with_top_k(self, query_service):
        """Test query with custom top_k."""
        request = QueryRequest(
            question="What is self-attention?",
            top_k=2,
        )
        response = await query_service.query(request)

        # Should return at most top_k chunks
        assert len(response.retrieved_chunks) <= 2

    @pytest.mark.asyncio
    async def test_query_with_paper_filter(self, sample_chunks):
        """Test query filtered to specific papers."""
        embedding = MockEmbeddingPort()
        vector_store = MockVectorStorePort(chunks=sample_chunks)
        llm = MockLLMPort()
        faithfulness = MockFaithfulnessPort()

        service = QueryService(
            embedding=embedding,
            vector_store=vector_store,
            llm=llm,
            faithfulness=faithfulness,
        )

        request = QueryRequest(
            question="What is self-attention?",
            paper_ids=["paper-001"],
        )
        response = await service.query(request)

        # All chunks should be from paper-001
        for chunk in response.retrieved_chunks:
            assert chunk.paper_id == "paper-001"

    @pytest.mark.asyncio
    async def test_query_stores_response(self, query_service):
        """Test that query stores response for later retrieval."""
        request = QueryRequest(question="What is self-attention?")
        response = await query_service.query(request)

        # Should be able to retrieve the stored response
        stored = await query_service.get_query(response.query_id)
        assert stored.query_id == response.query_id
        assert stored.question == response.question

    @pytest.mark.asyncio
    async def test_get_query_not_found(self, query_service):
        """Test that get_query raises error for unknown ID."""
        from src.domain.ports.query_storage import QueryNotFoundError

        with pytest.raises(QueryNotFoundError):
            await query_service.get_query("nonexistent-id")


class TestQueryResponseSchema:
    """Test query response schema validation."""

    @pytest.mark.asyncio
    async def test_response_has_all_required_fields(self, query_service):
        """Test that response contains all required fields."""
        request = QueryRequest(question="Test question")
        response = await query_service.query(request)

        # Check all required fields are present
        assert hasattr(response, "query_id")
        assert hasattr(response, "question")
        assert hasattr(response, "answer")
        assert hasattr(response, "citations")
        assert hasattr(response, "retrieved_chunks")
        assert hasattr(response, "faithfulness")
        assert hasattr(response, "trace")

    @pytest.mark.asyncio
    async def test_retrieved_chunk_schema(self, query_service):
        """Test that retrieved chunks have correct schema."""
        request = QueryRequest(question="Test question")
        response = await query_service.query(request)

        for chunk in response.retrieved_chunks:
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "paper_id")
            assert hasattr(chunk, "paper_title")
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "similarity_score")
            assert hasattr(chunk, "rerank_score")
            assert hasattr(chunk, "rank")

    @pytest.mark.asyncio
    async def test_trace_schema(self, query_service):
        """Test that trace has correct schema."""
        request = QueryRequest(question="Test question")
        response = await query_service.query(request)

        trace = response.trace
        assert hasattr(trace, "embedding_time_ms")
        assert hasattr(trace, "retrieval_time_ms")
        assert hasattr(trace, "reranking_time_ms")
        assert hasattr(trace, "generation_time_ms")
        assert hasattr(trace, "faithfulness_time_ms")
        assert hasattr(trace, "total_time_ms")


class TestEmptyResults:
    """Test handling of empty results."""

    @pytest.mark.asyncio
    async def test_query_with_no_chunks(self):
        """Test query when no chunks are available."""
        embedding = MockEmbeddingPort()
        vector_store = MockVectorStorePort(chunks=[])  # Empty
        llm = MockLLMPort()
        faithfulness = MockFaithfulnessPort()

        service = QueryService(
            embedding=embedding,
            vector_store=vector_store,
            llm=llm,
            faithfulness=faithfulness,
        )

        request = QueryRequest(question="What is self-attention?")
        response = await service.query(request)

        # Should return insufficient context response
        assert "cannot answer" in response.answer.lower()
        assert response.retrieved_chunks == []
        assert response.citations == []

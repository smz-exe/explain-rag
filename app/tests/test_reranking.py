"""Tests for reranking functionality."""

import pytest

from src.application.query_service import QueryService
from src.domain.entities.query import QueryRequest
from tests.conftest import (
    MockEmbeddingPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockRerankerPort,
    MockVectorStorePort,
)


@pytest.fixture
def query_service_with_reranking(
    sample_chunks,
) -> tuple[QueryService, MockRerankerPort]:
    """Create a QueryService with mock reranker."""
    embedding = MockEmbeddingPort()
    vector_store = MockVectorStorePort(chunks=sample_chunks)
    llm = MockLLMPort()
    faithfulness = MockFaithfulnessPort()
    reranker = MockRerankerPort(reverse_order=True)

    service = QueryService(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        faithfulness=faithfulness,
        reranker=reranker,
    )
    return service, reranker


@pytest.fixture
def query_service_without_reranking(sample_chunks) -> QueryService:
    """Create a QueryService without reranker."""
    embedding = MockEmbeddingPort()
    vector_store = MockVectorStorePort(chunks=sample_chunks)
    llm = MockLLMPort()
    faithfulness = MockFaithfulnessPort()

    return QueryService(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        faithfulness=faithfulness,
        reranker=None,
    )


class TestReranking:
    """Test reranking functionality in QueryService."""

    @pytest.mark.asyncio
    async def test_rerank_changes_order(self, query_service_with_reranking, sample_chunks):
        """Test that reranking changes the order of chunks."""
        service, reranker = query_service_with_reranking

        # Query with reranking enabled
        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # Reranker should have been called
        assert len(reranker.rerank_calls) == 1

        # Check that order is reversed (mock reranker reverses order)
        # Original order: chunk-001, chunk-002, chunk-003
        # Reranked order: chunk-003, chunk-002, chunk-001
        assert response.retrieved_chunks[0].chunk_id == "chunk-003"
        assert response.retrieved_chunks[1].chunk_id == "chunk-002"
        assert response.retrieved_chunks[2].chunk_id == "chunk-001"

    @pytest.mark.asyncio
    async def test_rerank_scores_populated(self, query_service_with_reranking):
        """Test that rerank_score is populated when reranking is enabled."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # All chunks should have rerank_score
        for chunk in response.retrieved_chunks:
            assert chunk.rerank_score is not None
            assert 0.0 <= chunk.rerank_score <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_disabled_no_effect(self, query_service_with_reranking, sample_chunks):
        """Test that reranking doesn't change order when disabled."""
        service, reranker = query_service_with_reranking

        # Query with reranking disabled
        request = QueryRequest(question="What is self-attention?", enable_reranking=False)
        response = await service.query(request)

        # Reranker should NOT have been called
        assert len(reranker.rerank_calls) == 0

        # Order should be original (based on similarity scores)
        assert response.retrieved_chunks[0].chunk_id == "chunk-001"
        assert response.retrieved_chunks[1].chunk_id == "chunk-002"
        assert response.retrieved_chunks[2].chunk_id == "chunk-003"

        # rerank_score should be None
        for chunk in response.retrieved_chunks:
            assert chunk.rerank_score is None

    @pytest.mark.asyncio
    async def test_rerank_without_reranker_configured(self, query_service_without_reranking):
        """Test that enabling reranking without reranker configured doesn't fail."""
        service = query_service_without_reranking

        # Query with reranking enabled but no reranker configured
        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # Should complete without error
        assert response.query_id is not None
        assert response.answer is not None

        # rerank_score should be None since no reranker
        for chunk in response.retrieved_chunks:
            assert chunk.rerank_score is None

    @pytest.mark.asyncio
    async def test_reranking_time_recorded(self, query_service_with_reranking):
        """Test that reranking time is recorded in trace."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # reranking_time_ms should be recorded
        assert response.trace.reranking_time_ms is not None
        assert response.trace.reranking_time_ms >= 0

    @pytest.mark.asyncio
    async def test_no_reranking_time_when_disabled(self, query_service_with_reranking):
        """Test that reranking time is None when reranking disabled."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=False)
        response = await service.query(request)

        # reranking_time_ms should be None
        assert response.trace.reranking_time_ms is None

    @pytest.mark.asyncio
    async def test_similarity_scores_preserved(self, query_service_with_reranking):
        """Test that original similarity scores are preserved after reranking."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # similarity_score should still be set (from original retrieval)
        for chunk in response.retrieved_chunks:
            assert chunk.similarity_score is not None
            assert 0.0 <= chunk.similarity_score <= 1.0


class TestCrossEncoderReranker:
    """Test the CrossEncoderReranker adapter."""

    def test_reranker_import(self):
        """Test that CrossEncoderReranker can be imported."""
        from src.adapters.outbound.cross_encoder_reranker import CrossEncoderReranker

        assert CrossEncoderReranker is not None

    def test_reranker_instantiation(self):
        """Test that CrossEncoderReranker can be instantiated."""
        from src.adapters.outbound.cross_encoder_reranker import CrossEncoderReranker

        # Should not load model until first use (lazy loading)
        reranker = CrossEncoderReranker()
        assert reranker._model is None
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_reranker_custom_model(self):
        """Test that CrossEncoderReranker accepts custom model name."""
        from src.adapters.outbound.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        assert reranker._model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"

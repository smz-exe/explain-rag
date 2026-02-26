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

    @pytest.mark.asyncio
    async def test_original_rank_preserved_with_reranking(self, query_service_with_reranking):
        """Test that original_rank reflects pre-reranking position."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        # Mock reranker reverses order: original [001, 002, 003] -> reranked [003, 002, 001]
        chunk_003 = next(c for c in response.retrieved_chunks if c.chunk_id == "chunk-003")
        chunk_002 = next(c for c in response.retrieved_chunks if c.chunk_id == "chunk-002")
        chunk_001 = next(c for c in response.retrieved_chunks if c.chunk_id == "chunk-001")

        # Original ranks (before reranking)
        assert chunk_001.original_rank == 1  # Was first in similarity search
        assert chunk_002.original_rank == 2
        assert chunk_003.original_rank == 3

        # Final ranks (after reranking reversed order)
        assert chunk_003.rank == 1  # Now first after reranking
        assert chunk_002.rank == 2
        assert chunk_001.rank == 3  # Now last after reranking

    @pytest.mark.asyncio
    async def test_original_rank_equals_rank_without_reranking(
        self, query_service_without_reranking
    ):
        """Test that original_rank equals rank when reranking is disabled."""
        service = query_service_without_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=False)
        response = await service.query(request)

        for chunk in response.retrieved_chunks:
            assert chunk.original_rank == chunk.rank

    @pytest.mark.asyncio
    async def test_rank_change_calculation(self, query_service_with_reranking):
        """Test that rank change can be calculated correctly."""
        service, _ = query_service_with_reranking

        request = QueryRequest(question="What is self-attention?", enable_reranking=True)
        response = await service.query(request)

        for chunk in response.retrieved_chunks:
            rank_change = chunk.original_rank - chunk.rank
            # Positive = promoted (moved up), Negative = demoted (moved down)
            if chunk.chunk_id == "chunk-003":
                assert rank_change == 2  # Was 3, now 1 (promoted by 2)
            elif chunk.chunk_id == "chunk-001":
                assert rank_change == -2  # Was 1, now 3 (demoted by 2)
            else:
                assert rank_change == 0  # chunk-002 stays at position 2


class TestFastEmbedReranker:
    """Test the FastEmbedReranker adapter."""

    def test_reranker_import(self):
        """Test that FastEmbedReranker can be imported."""
        from src.adapters.outbound.fastembed_reranker import FastEmbedReranker

        assert FastEmbedReranker is not None

    def test_reranker_instantiation(self):
        """Test that FastEmbedReranker can be instantiated."""
        from src.adapters.outbound.fastembed_reranker import FastEmbedReranker

        # Should not load model until first use (lazy loading)
        reranker = FastEmbedReranker()
        assert reranker._model is None
        assert reranker._model_name == "Xenova/ms-marco-MiniLM-L-6-v2"

    def test_reranker_custom_model(self):
        """Test that FastEmbedReranker accepts custom model name."""
        from src.adapters.outbound.fastembed_reranker import FastEmbedReranker

        reranker = FastEmbedReranker(model_name="Xenova/ms-marco-TinyBERT-L-2-v2")
        assert reranker._model_name == "Xenova/ms-marco-TinyBERT-L-2-v2"

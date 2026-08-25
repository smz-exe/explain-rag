"""Tests for query storage functionality."""

import pytest

from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult
from src.domain.entities.query import QueryResponse
from tests.conftest import MockQueryStoragePort


@pytest.fixture
def sample_query_response() -> QueryResponse:
    """Create a sample QueryResponse for testing."""
    return QueryResponse(
        query_id="test-query-001",
        question="What is self-attention?",
        answer="Self-attention is a mechanism [1].",
        citations=[],
        retrieved_chunks=[],
        faithfulness=FaithfulnessResult(score=0.9, claims=[]),
        trace=ExplanationTrace(
            embedding_time_ms=10.0,
            retrieval_time_ms=20.0,
            reranking_time_ms=None,
            generation_time_ms=100.0,
            faithfulness_time_ms=50.0,
            total_time_ms=180.0,
        ),
    )


class TestMockQueryStorage:
    """Test the mock query storage adapter."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, sample_query_response):
        """Test storing and retrieving a query."""
        storage = MockQueryStoragePort()

        await storage.store(sample_query_response)
        retrieved = await storage.get(sample_query_response.query_id)

        assert retrieved is not None
        assert retrieved.query_id == sample_query_response.query_id
        assert retrieved.question == sample_query_response.question

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test retrieving a non-existent query returns None."""
        storage = MockQueryStoragePort()

        result = await storage.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_recent(self, sample_query_response):
        """Test listing recent queries."""
        storage = MockQueryStoragePort()

        await storage.store(sample_query_response)
        recent = await storage.list_recent(limit=10)

        assert len(recent) == 1
        assert recent[0]["query_id"] == sample_query_response.query_id

    @pytest.mark.asyncio
    async def test_delete(self, sample_query_response):
        """Test deleting a query."""
        storage = MockQueryStoragePort()

        await storage.store(sample_query_response)
        deleted = await storage.delete(sample_query_response.query_id)

        assert deleted is True
        assert await storage.get(sample_query_response.query_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting a non-existent query returns False."""
        storage = MockQueryStoragePort()

        deleted = await storage.delete("nonexistent-id")
        assert deleted is False

"""Tests for query storage functionality."""

import tempfile
from pathlib import Path

import pytest

from src.adapters.outbound.sqlite_query_storage import SQLiteQueryStorage
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


class TestSQLiteQueryStorage:
    """Test the SQLite query storage adapter."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_queries.db"

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, temp_db_path, sample_query_response):
        """Test storing and retrieving a query in SQLite."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        await storage.store(sample_query_response)
        retrieved = await storage.get(sample_query_response.query_id)

        assert retrieved is not None
        assert retrieved.query_id == sample_query_response.query_id
        assert retrieved.question == sample_query_response.question
        assert retrieved.answer == sample_query_response.answer

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, temp_db_path):
        """Test retrieving a non-existent query returns None."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        result = await storage.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_recent(self, temp_db_path, sample_query_response):
        """Test listing recent queries from SQLite."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        await storage.store(sample_query_response)
        recent = await storage.list_recent(limit=10)

        assert len(recent) == 1
        assert recent[0]["query_id"] == sample_query_response.query_id
        assert recent[0]["question"] == sample_query_response.question
        assert "created_at" in recent[0]

    @pytest.mark.asyncio
    async def test_list_recent_limit(self, temp_db_path):
        """Test that list_recent respects the limit."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        # Store multiple queries
        for i in range(5):
            response = QueryResponse(
                query_id=f"query-{i}",
                question=f"Question {i}?",
                answer=f"Answer {i}.",
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
            await storage.store(response)

        recent = await storage.list_recent(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_delete(self, temp_db_path, sample_query_response):
        """Test deleting a query from SQLite."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        await storage.store(sample_query_response)
        deleted = await storage.delete(sample_query_response.query_id)

        assert deleted is True
        assert await storage.get(sample_query_response.query_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_db_path):
        """Test deleting a non-existent query returns False."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        deleted = await storage.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path, sample_query_response):
        """Test that data persists across storage instances."""
        # Store with first instance
        storage1 = SQLiteQueryStorage(db_path=temp_db_path)
        await storage1.store(sample_query_response)

        # Retrieve with second instance
        storage2 = SQLiteQueryStorage(db_path=temp_db_path)
        retrieved = await storage2.get(sample_query_response.query_id)

        assert retrieved is not None
        assert retrieved.query_id == sample_query_response.query_id

    @pytest.mark.asyncio
    async def test_upsert(self, temp_db_path, sample_query_response):
        """Test that storing the same query twice updates it."""
        storage = SQLiteQueryStorage(db_path=temp_db_path)

        await storage.store(sample_query_response)

        # Update the answer
        updated_response = QueryResponse(
            query_id=sample_query_response.query_id,
            question=sample_query_response.question,
            answer="Updated answer.",
            citations=[],
            retrieved_chunks=[],
            faithfulness=sample_query_response.faithfulness,
            trace=sample_query_response.trace,
        )
        await storage.store(updated_response)

        retrieved = await storage.get(sample_query_response.query_id)
        assert retrieved.answer == "Updated answer."

        # Should still only have one entry
        recent = await storage.list_recent(limit=10)
        assert len(recent) == 1

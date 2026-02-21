"""Tests for paper deletion functionality."""

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import MockVectorStorePort


class TestDeletePaperEndpoint:
    """Test the DELETE /papers/{paper_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_paper_not_found_endpoint(self, app):
        """Test DELETE endpoint returns 404 for non-existent paper."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # The real app's vector store is empty, so any paper_id returns 404
            response = await client.delete("/papers/nonexistent-paper-id")
            assert response.status_code == 404
            data = response.json()
            assert "Paper not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_delete_paper_returns_chunk_count(self, sample_chunks):
        """Test that deletion returns the count of deleted chunks."""
        mock_store = MockVectorStorePort(chunks=sample_chunks)

        # Delete the paper
        deleted_count = await mock_store.delete_paper("paper-001")

        assert deleted_count == 3  # We have 3 sample chunks
        assert len(mock_store.chunks) == 0

    @pytest.mark.asyncio
    async def test_delete_paper_idempotent(self, sample_chunks):
        """Test that deleting twice returns 0 on second attempt."""
        mock_store = MockVectorStorePort(chunks=sample_chunks)

        # First deletion
        first_count = await mock_store.delete_paper("paper-001")
        assert first_count == 3

        # Second deletion (should return 0)
        second_count = await mock_store.delete_paper("paper-001")
        assert second_count == 0

    @pytest.mark.asyncio
    async def test_delete_paper_not_found(self, sample_chunks):
        """Test deleting a non-existent paper returns 0."""
        mock_store = MockVectorStorePort(chunks=sample_chunks)

        deleted_count = await mock_store.delete_paper("nonexistent-paper")
        assert deleted_count == 0
        # Original chunks should remain
        assert len(mock_store.chunks) == 3


class TestDeletePaperRouter:
    """Test DELETE endpoint via router."""

    @pytest.mark.asyncio
    async def test_delete_endpoint_not_found(self, client):
        """Test DELETE returns 404 for unknown paper."""
        response = await client.delete("/papers/unknown-paper-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_endpoint_format(self, client):
        """Test the response format for 404."""
        response = await client.delete("/papers/unknown")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

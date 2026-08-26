"""Tests for paper deletion functionality."""

import pytest

from tests.conftest import MockVectorStorePort


class TestDeletePaperEndpoint:
    """Test the DELETE /papers/{paper_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_paper_requires_auth(self, client):
        """Test DELETE endpoint requires authentication."""
        response = await client.delete("/papers/nonexistent-paper-id")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_paper_not_found_endpoint(self, authenticated_client):
        """Test DELETE endpoint returns 404 for non-existent paper."""
        # The real app's vector store is empty, so any paper_id returns 404
        response = await authenticated_client.delete("/papers/nonexistent-paper-id")
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
    async def test_delete_paper_is_idempotent_and_reports_absence(self, sample_chunks):
        """Deleting twice: chunks removed, then "no such paper".

        0 and None are different answers — 0 means the paper existed and had no
        chunks, which used to be indistinguishable from not finding it at all.
        """
        mock_store = MockVectorStorePort(chunks=sample_chunks)

        assert await mock_store.delete_paper("paper-001") == 3
        assert await mock_store.delete_paper("paper-001") is None

    @pytest.mark.asyncio
    async def test_delete_paper_not_found(self, sample_chunks):
        """Deleting an unknown paper reports absence, not an empty deletion."""
        mock_store = MockVectorStorePort(chunks=sample_chunks)

        assert await mock_store.delete_paper("nonexistent-paper") is None
        # Original chunks should remain
        assert len(mock_store.chunks) == 3


class TestDeletePaperRouter:
    """Test DELETE endpoint via router."""

    @pytest.mark.asyncio
    async def test_delete_endpoint_not_found(self, authenticated_client):
        """Test DELETE returns 404 for unknown paper."""
        response = await authenticated_client.delete("/papers/unknown-paper-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_endpoint_format(self, authenticated_client):
        """Test the response format for 404."""
        response = await authenticated_client.delete("/papers/unknown")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

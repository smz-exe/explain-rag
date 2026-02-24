"""Tests for the stats endpoint."""

import pytest


@pytest.mark.asyncio
async def test_stats_endpoint_returns_correct_structure(client):
    """Test that /stats returns the expected structure."""
    response = await client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert "papers_count" in data
    assert "chunks_count" in data
    assert "queries_count" in data
    assert "backend_status" in data
    assert data["backend_status"] == "healthy"


@pytest.mark.asyncio
async def test_stats_returns_integer_counts(client):
    """Test that stats counts are integers."""
    response = await client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["papers_count"], int)
    assert isinstance(data["chunks_count"], int)
    assert isinstance(data["queries_count"], int)

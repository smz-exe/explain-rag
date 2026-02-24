"""Tests for the stats endpoint."""

import pytest


@pytest.mark.asyncio
async def test_stats_endpoint_requires_auth(client):
    """Test that /stats requires authentication."""
    response = await client.get("/stats")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_stats_endpoint_returns_correct_structure(authenticated_client):
    """Test that /stats returns the expected structure."""
    response = await authenticated_client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert "papers_count" in data
    assert "chunks_count" in data
    assert "queries_count" in data
    assert "backend_status" in data
    assert data["backend_status"] == "healthy"


@pytest.mark.asyncio
async def test_stats_returns_integer_counts(authenticated_client):
    """Test that stats counts are integers."""
    response = await authenticated_client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["papers_count"], int)
    assert isinstance(data["chunks_count"], int)
    assert isinstance(data["queries_count"], int)

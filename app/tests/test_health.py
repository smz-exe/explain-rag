import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test the health endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "papers_count" in data
    assert "chunks_count" in data


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test the root endpoint returns app info."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "ExplainRAG"
    assert "version" in data


@pytest.mark.asyncio
async def test_papers_endpoint_empty(client):
    """Test the papers endpoint returns empty list initially."""
    response = await client.get("/papers")
    assert response.status_code == 200

    data = response.json()
    assert "papers" in data
    assert "total" in data
    assert isinstance(data["papers"], list)

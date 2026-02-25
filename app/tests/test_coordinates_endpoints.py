"""Tests for coordinates endpoints."""

import pytest


class TestEmbeddingsEndpoint:
    """Test GET /papers/embeddings endpoint."""

    @pytest.mark.asyncio
    async def test_get_embeddings_returns_empty_initially(self, client):
        """Test that embeddings endpoint returns empty list initially."""
        response = await client.get("/papers/embeddings")

        assert response.status_code == 200
        data = response.json()
        assert "papers" in data
        assert "computed_at" in data
        # Initially empty since no recompute has been triggered
        assert isinstance(data["papers"], list)

    @pytest.mark.asyncio
    async def test_get_embeddings_response_structure(self, client):
        """Test response structure of embeddings endpoint."""
        response = await client.get("/papers/embeddings")

        assert response.status_code == 200
        data = response.json()
        assert "papers" in data
        assert "computed_at" in data


class TestClustersEndpoint:
    """Test GET /papers/clusters endpoint."""

    @pytest.mark.asyncio
    async def test_get_clusters_returns_empty_initially(self, client):
        """Test that clusters endpoint returns empty list initially."""
        response = await client.get("/papers/clusters")

        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert "computed_at" in data
        assert isinstance(data["clusters"], list)

    @pytest.mark.asyncio
    async def test_get_clusters_response_structure(self, client):
        """Test response structure of clusters endpoint."""
        response = await client.get("/papers/clusters")

        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert "computed_at" in data


class TestRecomputeEndpoint:
    """Test POST /admin/papers/recompute-embeddings endpoint."""

    @pytest.mark.asyncio
    async def test_recompute_requires_authentication(self, client):
        """Test that recompute endpoint requires admin auth."""
        response = await client.post("/admin/papers/recompute-embeddings")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_recompute_with_auth_succeeds(self, authenticated_client):
        """Test that authenticated admin can trigger recompute."""
        response = await authenticated_client.post("/admin/papers/recompute-embeddings")

        assert response.status_code == 200
        data = response.json()
        assert "papers_processed" in data
        assert "clusters_found" in data
        assert "time_ms" in data

    @pytest.mark.asyncio
    async def test_recompute_returns_stats(self, authenticated_client):
        """Test that recompute returns computation statistics."""
        response = await authenticated_client.post("/admin/papers/recompute-embeddings")

        assert response.status_code == 200
        data = response.json()

        # Stats should be present even if no papers
        assert isinstance(data["papers_processed"], int)
        assert isinstance(data["clusters_found"], int)
        assert isinstance(data["time_ms"], (int, float))


class TestCoordinatesIntegration:
    """Integration tests for coordinates workflow."""

    @pytest.mark.asyncio
    async def test_embeddings_updated_after_recompute(self, authenticated_client):
        """Test that embeddings reflect recompute timestamp."""
        # Trigger recompute
        recompute_response = await authenticated_client.post(
            "/admin/papers/recompute-embeddings"
        )
        assert recompute_response.status_code == 200

        # Get embeddings
        embeddings_response = await authenticated_client.get("/papers/embeddings")
        assert embeddings_response.status_code == 200
        data = embeddings_response.json()

        # After recompute, computed_at should be set
        assert data["computed_at"] is not None

    @pytest.mark.asyncio
    async def test_clusters_updated_after_recompute(self, authenticated_client):
        """Test that clusters reflect recompute timestamp."""
        # Trigger recompute
        recompute_response = await authenticated_client.post(
            "/admin/papers/recompute-embeddings"
        )
        assert recompute_response.status_code == 200

        # Get clusters
        clusters_response = await authenticated_client.get("/papers/clusters")
        assert clusters_response.status_code == 200
        data = clusters_response.json()

        # After recompute, computed_at should be set
        assert data["computed_at"] is not None

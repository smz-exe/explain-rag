"""Tests for security headers, request body size limit, and liveness/readiness probes."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app
from tests.conftest import (
    MockClusteringPort,
    MockCoordinatesStoragePort,
    MockDimensionalityReductionPort,
    MockEmbeddingPort,
    MockEvaluationPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockRerankerPort,
    MockVectorStorePort,
)


def build_app(sample_chunks, vector_store=None):
    """Create a test app, optionally with a custom vector store."""
    return create_app(
        embedding=MockEmbeddingPort(),
        vector_store=vector_store or MockVectorStorePort(chunks=sample_chunks),
        llm=MockLLMPort(),
        faithfulness=MockFaithfulnessPort(),
        reranker=MockRerankerPort(),
        evaluator=MockEvaluationPort(),
        query_storage=MockQueryStoragePort(),
        coordinates_storage=MockCoordinatesStoragePort(),
        dim_reducer=MockDimensionalityReductionPort(),
        clusterer=MockClusteringPort(),
    )


class FailingVectorStore(MockVectorStorePort):
    """Vector store whose stats call fails, simulating an unreachable database."""

    async def get_stats(self) -> dict:
        raise ConnectionError("database unreachable")


class TestSecurityHeaders:
    """All responses must carry baseline security headers."""

    async def test_headers_present_on_responses(self, client):
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"

    async def test_hsts_absent_in_development(self, client):
        response = await client.get("/health")

        assert "strict-transport-security" not in response.headers

    async def test_hsts_present_in_production(self, monkeypatch, sample_chunks):
        monkeypatch.setenv("ENVIRONMENT", "production")
        app = build_app(sample_chunks)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/health")

        assert response.headers["strict-transport-security"] == (
            "max-age=63072000; includeSubDomains"
        )

    async def test_headers_present_on_error_responses(self, client):
        response = await client.get("/query/nonexistent-id/explanation")

        assert response.status_code == 404
        assert response.headers["x-content-type-options"] == "nosniff"


class TestBodySizeLimit:
    """Requests larger than the configured limit are rejected with 413."""

    @pytest.fixture
    async def small_limit_client(self, monkeypatch, sample_chunks):
        monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "200")
        app = build_app(sample_chunks)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_oversized_body_rejected(self, small_limit_client):
        payload = {"question": "x" * 500}

        response = await small_limit_client.post("/query", json=payload)

        assert response.status_code == 413
        assert response.json()["error"] == "request_too_large"

    async def test_body_within_limit_accepted(self, small_limit_client):
        response = await small_limit_client.post("/query", json={"question": "Hi?"})

        assert response.status_code == 200

    async def test_default_limit_allows_normal_queries(self, client):
        response = await client.post("/query", json={"question": "What is self-attention?"})

        assert response.status_code == 200


class TestProbes:
    """Liveness and readiness endpoints for orchestration health checks."""

    async def test_live_returns_alive(self, client):
        response = await client.get("/live")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    async def test_ready_when_database_reachable(self, client):
        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    async def test_ready_returns_503_when_database_unreachable(self, sample_chunks):
        app = build_app(sample_chunks, vector_store=FailingVectorStore())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}

"""Tests for the /ingest HTTP endpoint using an injected fake paper source."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.adapters.inbound.http.auth import require_admin
from src.adapters.inbound.http.ingest import create_router
from src.application.ingestion_service import IngestionService
from tests.test_ingestion_service import (
    FakeEmbedding,
    FakePaperSource,
    RecordingVectorStore,
)


@pytest.fixture
def ingest_app(sample_paper, sample_chunks) -> FastAPI:
    """Minimal app mounting only the ingest router with fake adapters."""
    service = IngestionService(
        paper_source=FakePaperSource(sample_paper, sample_chunks),
        embedding=FakeEmbedding(),
        vector_store=RecordingVectorStore(),
    )
    app = FastAPI()
    app.include_router(create_router(service))
    app.dependency_overrides[require_admin] = lambda: None
    return app


@pytest.fixture
async def ingest_client(ingest_app):
    """Async client for the ingest-only app (admin dependency overridden)."""
    async with AsyncClient(transport=ASGITransport(app=ingest_app), base_url="http://test") as c:
        yield c


class TestIngestEndpoint:
    """Tests for POST /ingest."""

    async def test_ingest_by_arxiv_ids(self, ingest_client, sample_paper, sample_chunks):
        response = await ingest_client.post("/ingest", json={"arxiv_ids": [sample_paper.arxiv_id]})

        assert response.status_code == 200
        body = response.json()
        assert len(body["ingested"]) == 1
        assert body["ingested"][0]["arxiv_id"] == sample_paper.arxiv_id
        assert body["ingested"][0]["chunk_count"] == len(sample_chunks)
        assert body["errors"] == []

    async def test_ingest_by_search_query(self, ingest_client, sample_paper):
        response = await ingest_client.post(
            "/ingest", json={"search_query": "attention mechanisms", "max_results": 1}
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["ingested"]) == 1
        assert body["ingested"][0]["arxiv_id"] == sample_paper.arxiv_id

    async def test_empty_request_returns_empty_response(self, ingest_client):
        response = await ingest_client.post("/ingest", json={})

        assert response.status_code == 200
        assert response.json() == {"ingested": [], "errors": []}

    async def test_requires_admin(self, client):
        """The real app must reject unauthenticated ingestion requests."""
        response = await client.post("/ingest", json={})

        assert response.status_code == 401

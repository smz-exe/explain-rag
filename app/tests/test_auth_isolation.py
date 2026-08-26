"""Tests that admin auth is bound to its own application instance.

require_admin used to read configuration from module-level globals assigned by
create_router(). Whichever application was constructed last therefore supplied
the JWT secret for every application in the process, so a token minted by one
app would be honoured by another — and an app built before the last one would
validate against a secret it never configured.
"""

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

SECRET_A = "secret-for-app-a-which-is-long-enough-32"
SECRET_B = "secret-for-app-b-which-is-different-000"


def build_app(sample_chunks, monkeypatch, jwt_secret: str):
    monkeypatch.setenv("JWT_SECRET_KEY", jwt_secret)
    return create_app(
        embedding=MockEmbeddingPort(),
        vector_store=MockVectorStorePort(chunks=sample_chunks),
        llm=MockLLMPort(),
        faithfulness=MockFaithfulnessPort(),
        reranker=MockRerankerPort(),
        evaluator=MockEvaluationPort(),
        query_storage=MockQueryStoragePort(),
        coordinates_storage=MockCoordinatesStoragePort(),
        dim_reducer=MockDimensionalityReductionPort(),
        clusterer=MockClusteringPort(),
    )


async def login_and_get_cookie(app) -> str:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/auth/login", json={"username": "admin", "password": "testpassword"}
        )
        assert response.status_code == 200, response.text
        return response.cookies["access_token"]


class TestAdminAuthIsScopedToItsApp:
    @pytest.mark.asyncio
    async def test_token_from_another_app_is_rejected(self, sample_chunks, monkeypatch):
        """A session minted under one secret must not unlock a differently keyed app."""
        app_a = build_app(sample_chunks, monkeypatch, SECRET_A)
        token_a = await login_and_get_cookie(app_a)

        # Built afterwards: with module-level globals, this app's secret would
        # have overwritten app_a's for everyone.
        app_b = build_app(sample_chunks, monkeypatch, SECRET_B)

        async with AsyncClient(
            transport=ASGITransport(app=app_b), base_url="http://test"
        ) as client:
            client.cookies.set("access_token", token_a)
            response = await client.get("/stats")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_earlier_app_still_accepts_its_own_token(self, sample_chunks, monkeypatch):
        """Building a second app must not invalidate the first app's sessions."""
        app_a = build_app(sample_chunks, monkeypatch, SECRET_A)
        token_a = await login_and_get_cookie(app_a)

        build_app(sample_chunks, monkeypatch, SECRET_B)  # last-constructed app

        async with AsyncClient(
            transport=ASGITransport(app=app_a), base_url="http://test"
        ) as client:
            client.cookies.set("access_token", token_a)
            response = await client.get("/stats")

        assert response.status_code == 200

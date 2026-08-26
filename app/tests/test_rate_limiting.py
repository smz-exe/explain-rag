"""Tests for rate limiting on the query endpoint.

Behind Fly's proxy every request arrives from the same socket peer, so keying
the limiter on request.client.host collapses "10/minute per IP" into one global
bucket: a single visitor could 429 the whole demo. These tests pin the key
derivation and the per-application isolation that makes it testable at all.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import Headers

from src.adapters.inbound.http.rate_limit import client_key_for
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


def build_app(sample_chunks, monkeypatch, **env):
    """Build an app with rate limiting on and a fresh limiter."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
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


class FakeRequest:
    """Minimal stand-in exposing what the key function reads.

    Uses Starlette's Headers so lookups are case-insensitive exactly as they
    are for a real request.
    """

    def __init__(self, headers: dict[str, str], peer: str | None = "10.0.0.1"):
        self.headers = Headers(headers)
        self.client = type("Client", (), {"host": peer})() if peer else None


class TestClientKeyDerivation:
    """The limiter key must identify the real caller, not the proxy."""

    def test_uses_socket_peer_when_no_header_configured(self):
        request = FakeRequest(headers={"fly-client-ip": "203.0.113.9"}, peer="10.0.0.1")

        assert client_key_for(request, client_ip_header=None) == "10.0.0.1"

    def test_prefers_configured_forwarded_header(self):
        request = FakeRequest(headers={"fly-client-ip": "203.0.113.9"}, peer="10.0.0.1")

        assert client_key_for(request, client_ip_header="Fly-Client-IP") == "203.0.113.9"

    def test_header_lookup_is_case_insensitive(self):
        request = FakeRequest(headers={"Fly-Client-IP": "203.0.113.9"}, peer="10.0.0.1")

        assert client_key_for(request, client_ip_header="fly-client-ip") == "203.0.113.9"

    def test_falls_back_to_peer_when_configured_header_absent(self):
        request = FakeRequest(headers={}, peer="10.0.0.1")

        assert client_key_for(request, client_ip_header="Fly-Client-IP") == "10.0.0.1"

    def test_distinct_clients_behind_one_proxy_get_distinct_keys(self):
        """The whole point: one proxy peer, two callers, two buckets."""
        first = FakeRequest(headers={"fly-client-ip": "203.0.113.9"}, peer="10.0.0.1")
        second = FakeRequest(headers={"fly-client-ip": "198.51.100.4"}, peer="10.0.0.1")

        assert client_key_for(first, client_ip_header="Fly-Client-IP") != client_key_for(
            second, client_ip_header="Fly-Client-IP"
        )


class TestQueryRateLimitEnforcement:
    """End-to-end: the limit actually fires, and only for the offending client."""

    @pytest.mark.asyncio
    async def test_exceeding_the_limit_returns_429(self, sample_chunks, monkeypatch):
        app = build_app(sample_chunks, monkeypatch, RATE_LIMIT_QUERY="2/minute")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"question": "What is self-attention?"}
            first = await client.post("/query", json=payload)
            second = await client.post("/query", json=payload)
            third = await client.post("/query", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
        # A 429 without Retry-After leaves clients guessing when to come back.
        assert third.headers["retry-after"] == "60"

    @pytest.mark.asyncio
    async def test_one_client_hitting_the_limit_does_not_block_another(
        self, sample_chunks, monkeypatch
    ):
        """Regression: with a proxy-derived key, the first client 429s everyone."""
        app = build_app(
            sample_chunks,
            monkeypatch,
            RATE_LIMIT_QUERY="1/minute",
            CLIENT_IP_HEADER="Fly-Client-IP",
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"question": "What is self-attention?"}
            noisy = {"Fly-Client-IP": "203.0.113.9"}
            other = {"Fly-Client-IP": "198.51.100.4"}

            assert (await client.post("/query", json=payload, headers=noisy)).status_code == 200
            assert (await client.post("/query", json=payload, headers=noisy)).status_code == 429

            # A different client must still be served.
            innocent = await client.post("/query", json=payload, headers=other)

        assert innocent.status_code == 200

    @pytest.mark.asyncio
    async def test_limiter_state_is_per_application(self, sample_chunks, monkeypatch):
        """A module-global limiter leaks counts between apps and makes tests order-dependent."""
        payload = {"question": "What is self-attention?"}
        first_app = build_app(sample_chunks, monkeypatch, RATE_LIMIT_QUERY="1/minute")

        async with AsyncClient(
            transport=ASGITransport(app=first_app), base_url="http://test"
        ) as client:
            assert (await client.post("/query", json=payload)).status_code == 200
            assert (await client.post("/query", json=payload)).status_code == 429

        second_app = build_app(sample_chunks, monkeypatch, RATE_LIMIT_QUERY="1/minute")
        async with AsyncClient(
            transport=ASGITransport(app=second_app), base_url="http://test"
        ) as client:
            fresh = await client.post("/query", json=payload)

        assert fresh.status_code == 200


class TestLoginRateLimit:
    """Online password guessing must be throttled too."""

    @pytest.mark.asyncio
    async def test_repeated_failed_logins_are_throttled(self, sample_chunks, monkeypatch):
        app = build_app(sample_chunks, monkeypatch, RATE_LIMIT_LOGIN="3/minute")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"username": "admin", "password": "wrong"}
            statuses = [
                (await client.post("/auth/login", json=payload)).status_code for _ in range(4)
            ]

        assert statuses[:3] == [401, 401, 401]
        assert statuses[3] == 429

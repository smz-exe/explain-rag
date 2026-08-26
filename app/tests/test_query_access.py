"""Tests for capability-token access control on stored queries.

A query_id used to be both the identifier and the access secret: anyone who
learned an id could read the full answer, and GET /query/list handed those ids
out to anonymous callers. These tests pin the split — the id identifies, a
short-lived signed token grants access.
"""

import jwt
import pytest

from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings instance matching the conftest test environment."""
    return Settings()


async def _submit_query(client) -> dict:
    """Submit a query and return the raw response payload."""
    response = await client.post("/query", json={"question": "What is self-attention?"})
    assert response.status_code == 200, response.text
    return response.json()


class TestShareTokenIssuance:
    """POST /query must hand the caller a capability for its own result."""

    @pytest.mark.asyncio
    async def test_query_response_includes_share_token(self, client):
        payload = await _submit_query(client)

        assert payload["share_token"], "POST /query must return a share_token"

    @pytest.mark.asyncio
    async def test_share_token_is_scoped_to_its_query(self, client, settings):
        payload = await _submit_query(client)

        claims = jwt.decode(
            payload["share_token"],
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        assert claims["qid"] == payload["query_id"]
        assert claims["scope"] == "query:read"
        assert "exp" in claims, "share tokens must expire"


class TestExplanationAccess:
    """GET /query/{id}/explanation requires a capability or admin."""

    @pytest.mark.asyncio
    async def test_rejects_request_without_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(f"/query/{payload['query_id']}/explanation")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_matching_share_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(
            f"/query/{payload['query_id']}/explanation",
            headers={"Authorization": f"Bearer {payload['share_token']}"},
        )

        assert response.status_code == 200
        assert response.json()["query_id"] == payload["query_id"]

    @pytest.mark.asyncio
    async def test_rejects_token_issued_for_another_query(self, client):
        first = await _submit_query(client)
        second = await _submit_query(client)
        assert first["query_id"] != second["query_id"]

        response = await client.get(
            f"/query/{second['query_id']}/explanation",
            headers={"Authorization": f"Bearer {first['share_token']}"},
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_tampered_token(self, client):
        payload = await _submit_query(client)
        tampered = payload["share_token"][:-2] + (
            "ab" if payload["share_token"][-2:] != "ab" else "cd"
        )

        response = await client.get(
            f"/query/{payload['query_id']}/explanation",
            headers={"Authorization": f"Bearer {tampered}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_expired_token(self, client, settings):
        payload = await _submit_query(client)
        expired = jwt.encode(
            {"qid": payload["query_id"], "scope": "query:read", "exp": 1_000_000_000},
            settings.jwt_secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )

        response = await client.get(
            f"/query/{payload['query_id']}/explanation",
            headers={"Authorization": f"Bearer {expired}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_admin_login_cookie_used_as_share_token(self, client, settings):
        """An auth JWT is not a capability: scope must be checked, not just the signature."""
        payload = await _submit_query(client)
        auth_shaped = jwt.encode(
            {"sub": "admin", "is_admin": True, "qid": payload["query_id"]},
            settings.jwt_secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )

        response = await client.get(
            f"/query/{payload['query_id']}/explanation",
            headers={"Authorization": f"Bearer {auth_shaped}"},
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_cookie_grants_access_without_share_token(
        self, authenticated_client, client
    ):
        payload = await _submit_query(client)

        response = await authenticated_client.get(f"/query/{payload['query_id']}/explanation")

        assert response.status_code == 200


class TestExportAccess:
    """GET /query/{id}/export is a full dump and needs the same gate."""

    @pytest.mark.asyncio
    async def test_rejects_request_without_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(f"/query/{payload['query_id']}/export")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_matching_share_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(
            f"/query/{payload['query_id']}/export",
            headers={"Authorization": f"Bearer {payload['share_token']}"},
        )

        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]


class TestFaithfulnessPollAccess:
    """The deferred-verification poll exposes the same report as explanation."""

    @pytest.mark.asyncio
    async def test_rejects_request_without_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(f"/query/{payload['query_id']}/faithfulness")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_matching_share_token(self, client):
        payload = await _submit_query(client)

        response = await client.get(
            f"/query/{payload['query_id']}/faithfulness",
            headers={"Authorization": f"Bearer {payload['share_token']}"},
        )

        assert response.status_code == 200
        assert response.json()["query_id"] == payload["query_id"]


class TestQueryListIsAdminOnly:
    """The id-enumeration path must be closed to anonymous callers."""

    @pytest.mark.asyncio
    async def test_anonymous_cannot_list_queries(self, client):
        response = await client.get("/query/list")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_share_token_does_not_grant_listing(self, client):
        payload = await _submit_query(client)

        response = await client.get(
            "/query/list",
            headers={"Authorization": f"Bearer {payload['share_token']}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_can_list_queries(self, authenticated_client):
        response = await authenticated_client.get("/query/list")

        assert response.status_code == 200
        assert "queries" in response.json()

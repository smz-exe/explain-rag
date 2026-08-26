"""Tests for authentication endpoints."""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import pytest

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

# Generate a test password hash
TEST_PASSWORD = "testpassword123"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt(12)).decode()
TEST_JWT_SECRET = "test-secret-key-at-least-32-chars"


def _mock_adapters() -> dict:
    """Adapters for create_app() so no real service is constructed."""
    return {
        "embedding": MockEmbeddingPort(),
        "vector_store": MockVectorStorePort(chunks=[]),
        "llm": MockLLMPort(),
        "faithfulness": MockFaithfulnessPort(),
        "reranker": MockRerankerPort(),
        "evaluator": MockEvaluationPort(),
        "query_storage": MockQueryStoragePort(),
        "coordinates_storage": MockCoordinatesStoragePort(),
        "dim_reducer": MockDimensionalityReductionPort(),
        "clusterer": MockClusteringPort(),
    }


class TestAuthEndpoints:
    """Tests for /auth/* endpoints."""

    @pytest.mark.asyncio
    async def test_login_invalid_username(self, client):
        """Test login with invalid username."""
        response = await client.post(
            "/auth/login",
            json={"username": "wronguser", "password": "anypassword"},
        )
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, client):
        """Test login with invalid password."""
        response = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"},
        )
        # Will fail because no password hash is configured in test
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_without_token(self, client):
        """Test /auth/me without authentication."""
        response = await client.get("/auth/me")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_logout(self, client):
        """Test logout endpoint."""
        response = await client.post("/auth/logout")
        assert response.status_code == 200
        assert response.json()["message"] == "Logged out"

    @pytest.mark.asyncio
    async def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = await client.post("/auth/login", json={"username": "admin"})
        assert response.status_code == 422  # Validation error


class TestAuthWithConfiguredPassword:
    """Tests with a properly configured admin password."""

    @pytest.fixture
    def configured_app(self, monkeypatch):
        """Create an app with a configured admin password."""
        monkeypatch.setenv("ADMIN_USERNAME", "testadmin")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", TEST_PASSWORD_HASH)
        monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)

        # Import after setting env vars
        from src.main import create_app

        # Inject mocks: without them create_app() builds the real Postgres,
        # Anthropic, and FastEmbed adapters and reads the developer's app/.env.
        return create_app(**_mock_adapters())

    @pytest.fixture
    async def configured_client(self, configured_app):
        """Create a test client with configured auth."""
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=configured_app),
            base_url="http://test",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_login_success(self, configured_client):
        """Test successful login."""
        response = await configured_client.post(
            "/auth/login",
            json={"username": "testadmin", "password": TEST_PASSWORD},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Login successful"
        assert "access_token" in response.cookies

    @pytest.mark.asyncio
    async def test_me_with_valid_token(self, configured_client):
        """Test /auth/me with valid token."""
        # First login
        login_response = await configured_client.post(
            "/auth/login",
            json={"username": "testadmin", "password": TEST_PASSWORD},
        )
        assert login_response.status_code == 200

        # Then get user info (cookies are automatically included)
        me_response = await configured_client.get("/auth/me")
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "testadmin"
        assert me_response.json()["is_admin"] is True

    @pytest.mark.asyncio
    async def test_full_auth_flow(self, configured_client):
        """Test full login -> me -> logout flow."""
        # Login
        login_response = await configured_client.post(
            "/auth/login",
            json={"username": "testadmin", "password": TEST_PASSWORD},
        )
        assert login_response.status_code == 200

        # Check auth
        me_response = await configured_client.get("/auth/me")
        assert me_response.status_code == 200

        # Logout
        logout_response = await configured_client.post("/auth/logout")
        assert logout_response.status_code == 200

        # The point of logging out: access is actually revoked.
        assert (await configured_client.get("/auth/me")).status_code == 401
        assert (await configured_client.get("/stats")).status_code == 401


class TestTokenRejectionPaths:
    """Every way a token can be unusable must be rejected.

    The production code has four distinct rejection branches (expired and
    otherwise-invalid, in both require_admin and /auth/me) plus the is_admin
    check, and none of them had a test.
    """

    @pytest.fixture
    def configured_app(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USERNAME", "testadmin")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", TEST_PASSWORD_HASH)
        monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)

        from src.main import create_app

        return create_app(**_mock_adapters())

    @pytest.fixture
    async def client_with(self, configured_app):
        from httpx import ASGITransport, AsyncClient

        async def _make(token: str | None) -> AsyncClient:
            c = AsyncClient(transport=ASGITransport(app=configured_app), base_url="http://test")
            if token is not None:
                c.cookies.set("access_token", token)
            return c

        return _make

    @staticmethod
    def _token(**claims) -> str:
        payload = {"sub": "testadmin", "is_admin": True}
        payload.update(claims)
        return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")

    @pytest.mark.asyncio
    async def test_expired_token_is_rejected(self, client_with):
        expired = self._token(exp=datetime.now(UTC) - timedelta(minutes=1))
        client = await client_with(expired)

        async with client:
            assert (await client.get("/stats")).status_code == 401
            me = await client.get("/auth/me")

        assert me.status_code == 401
        assert me.json()["detail"] == "Token expired"

    @pytest.mark.asyncio
    async def test_token_signed_with_another_key_is_rejected(self, client_with):
        forged = jwt.encode(
            {"sub": "testadmin", "is_admin": True},
            "an-entirely-different-secret-key-32-chars",
            algorithm="HS256",
        )
        client = await client_with(forged)

        async with client:
            assert (await client.get("/stats")).status_code == 401
            assert (await client.get("/auth/me")).status_code == 401

    @pytest.mark.asyncio
    async def test_garbage_token_is_rejected(self, client_with):
        client = await client_with("not-a-jwt-at-all")

        async with client:
            assert (await client.get("/stats")).status_code == 401
            assert (await client.get("/auth/me")).status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_without_admin_flag_is_forbidden(self, client_with):
        """Authenticated but not authorized: 403, not 401."""
        client = await client_with(self._token(is_admin=False))

        async with client:
            response = await client.get("/stats")

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_token_without_subject_is_rejected(self, client_with):
        client = await client_with(self._token(sub=None))

        async with client:
            assert (await client.get("/stats")).status_code == 401

    @pytest.mark.asyncio
    async def test_token_for_an_unknown_user_is_rejected_by_me(self, client_with):
        """/auth/me resolves the subject against storage, unlike require_admin."""
        client = await client_with(self._token(sub="someone-else"))

        async with client:
            response = await client.get("/auth/me")

        assert response.status_code == 401
        assert response.json()["detail"] == "User not found"


class TestCookieAttributes:
    """Cookie attributes must match the deployment environment.

    The frontend (vercel.app) and API (fly.dev) are different sites, so the
    auth cookie is third-party from the browser's perspective. In production
    it must be SameSite=None; Secure; Partitioned (CHIPS) to survive
    third-party-cookie blocking.
    """

    @pytest.fixture
    def make_client(self, monkeypatch):
        """Build a client factory parameterized by ENVIRONMENT."""
        monkeypatch.setenv("ADMIN_USERNAME", "testadmin")
        monkeypatch.setenv("ADMIN_PASSWORD_HASH", TEST_PASSWORD_HASH)
        monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)

        from httpx import ASGITransport, AsyncClient

        def _make(environment: str) -> AsyncClient:
            monkeypatch.setenv("ENVIRONMENT", environment)
            from src.main import create_app

            app = create_app(**_mock_adapters())
            return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

        return _make

    @pytest.mark.asyncio
    async def test_production_login_cookie_is_partitioned(self, make_client):
        """Production cookies must carry SameSite=None; Secure; Partitioned."""
        async with make_client("production") as client:
            response = await client.post(
                "/auth/login",
                json={"username": "testadmin", "password": TEST_PASSWORD},
            )

        assert response.status_code == 200
        set_cookie = response.headers["set-cookie"].lower()
        assert "samesite=none" in set_cookie
        assert "secure" in set_cookie
        assert "partitioned" in set_cookie
        assert "httponly" in set_cookie

    @pytest.mark.asyncio
    async def test_development_login_cookie_is_not_partitioned(self, make_client):
        """Development cookies stay Lax without Secure/Partitioned (plain-http localhost)."""
        async with make_client("development") as client:
            response = await client.post(
                "/auth/login",
                json={"username": "testadmin", "password": TEST_PASSWORD},
            )

        assert response.status_code == 200
        set_cookie = response.headers["set-cookie"].lower()
        assert "samesite=lax" in set_cookie
        assert "partitioned" not in set_cookie

    @pytest.mark.asyncio
    async def test_production_logout_deletes_partitioned_cookie(self, make_client):
        """Logout must delete with matching attributes, or browsers keep the cookie."""
        async with make_client("production") as client:
            response = await client.post("/auth/logout")

        assert response.status_code == 200
        set_cookie = response.headers["set-cookie"].lower()
        assert 'access_token=""' in set_cookie or "access_token=;" in set_cookie
        assert "max-age=0" in set_cookie
        assert "samesite=none" in set_cookie
        assert "partitioned" in set_cookie

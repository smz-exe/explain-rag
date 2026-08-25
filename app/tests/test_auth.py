"""Tests for authentication endpoints."""

import bcrypt
import pytest

# Generate a test password hash
TEST_PASSWORD = "testpassword123"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt(12)).decode()


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
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-32-chars")

        # Import after setting env vars
        from src.main import create_app

        return create_app()

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

        # Should no longer be authenticated after logout
        # (Note: In a real browser, the cookie would be deleted)


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
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-at-least-32-chars")

        from httpx import ASGITransport, AsyncClient

        def _make(environment: str) -> AsyncClient:
            monkeypatch.setenv("ENVIRONMENT", environment)
            from src.main import create_app

            app = create_app()
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

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app


@pytest.fixture
def app():
    """Create a test application instance."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

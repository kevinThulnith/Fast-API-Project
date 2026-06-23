import pytest
from httpx import ASGITransport, AsyncClient

from main import app

# ---------------------------------------------------------------------------
# Shared fixtures for the whole test suite (tests/, tests_generated/, etc.)
#
# pytest auto-discovers conftest.py files and makes their fixtures available
# to every test module at or below the directory this file lives in — no
# import needed in the test files themselves. This file should live at the
# project root (next to main.py) so it covers both tests/ and tests_generated/.
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Async HTTP client for the FastAPI app using ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
async def reset_db(client):
    """Reset the database before each test to ensure isolation."""
    response = await client.post("/api/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"
    # Return the user/post counts for optional checks
    return data

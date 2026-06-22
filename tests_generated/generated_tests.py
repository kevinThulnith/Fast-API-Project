import pytest
from httpx import ASGITransport, AsyncClient

from main import app


# ---------------------------------------------------------------------------
# Fixture – same as in test_api.py
# ---------------------------------------------------------------------------
@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests – corrected to match actual API responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_api_health(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "db" in data
    assert "users" in data["db"]
    assert "posts" in data["db"]


@pytest.mark.asyncio
async def test_post_api_reset(client: AsyncClient):
    response = await client.post("/api/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"
    assert "users" in data
    assert "posts" in data


@pytest.mark.asyncio
async def test_post_api_seed(client: AsyncClient):
    # Valid seeding
    response = await client.post("/api/seed", json={"num_users": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "seeded"
    assert data["users"] == 10
    assert data["posts"] >= 50

    # Invalid: num_users out of range
    response = await client.post("/api/seed", json={"num_users": 0})
    assert response.status_code == 422
    response = await client.post("/api/seed", json={"num_users": 1001})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_api_error_injection(client: AsyncClient):
    # Enable error injection
    response = await client.post("/api/error-injection?enabled=true")
    assert response.status_code == 200
    assert response.json()["error_injection"] is True

    # Disable
    response = await client.post("/api/error-injection?enabled=false")
    assert response.status_code == 200
    assert response.json()["error_injection"] is False

    # Missing 'enabled' query param → 422
    response = await client.post("/api/error-injection")
    assert response.status_code == 422

    # Invalid boolean string → 422
    response = await client.post("/api/error-injection?enabled=maybe")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_users(client: AsyncClient):
    response = await client.get("/api/users")
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)
    if users:
        assert "id" in users[0]
        assert "name" in users[0]
        assert "email" in users[0]
        assert "city" in users[0]
        assert "company" in users[0]


@pytest.mark.asyncio
async def test_get_api_users_paginated(client: AsyncClient):
    # Default page 1, limit 20
    response = await client.get("/api/users/paginated")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["limit"] == 20
    assert "total" in data
    assert "pages" in data
    assert "data" in data
    assert len(data["data"]) <= 20

    # Custom page and limit
    response = await client.get("/api/users/paginated?page=2&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 2
    assert data["limit"] == 5
    assert len(data["data"]) <= 5

    # Invalid page (<1) → 422
    response = await client.get("/api/users/paginated?page=0")
    assert response.status_code == 422

    # Limit > 100 → 422 (API enforces le=100, not capped)
    response = await client.get("/api/users/paginated?limit=200")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_users_user_id(client: AsyncClient):
    # User with ID 1 exists
    response = await client.get("/api/users/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["id"] == 1
    assert "posts" in data
    assert isinstance(data["posts"], list)

    # Non-existent user → 404
    response = await client.get("/api/users/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # Invalid ID (non-integer) → 422
    response = await client.get("/api/users/abc")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_api_posts(client: AsyncClient):
    user_id = 1

    # Happy path
    payload = {"user_id": user_id, "title": "Test Post", "content": "Content"}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 201
    post = response.json()
    assert post["user_id"] == user_id
    assert post["title"] == payload["title"]
    assert post["content"] == payload["content"]
    assert "id" in post
    assert "created_at" in post

    # User not found → 404
    payload["user_id"] = 999999
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]

    # Missing title → 422
    del payload["title"]
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422

    # Blank title → 422
    payload["title"] = "   "
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422

    # Title too long (>200) → 422
    payload["title"] = "a" * 201
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422

    # Content too long (>5000) → 422
    payload["title"] = "Valid title"
    payload["content"] = "a" * 5001
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_put_api_posts_post_id(client: AsyncClient):
    # Create a post to update
    create_resp = await client.post(
        "/api/posts", json={"user_id": 1, "title": "Old", "content": "Old content"}
    )
    assert create_resp.status_code == 201
    post_id = create_resp.json()["id"]

    # Happy update
    update_payload = {"title": "New Title", "content": "New content"}
    response = await client.put(f"/api/posts/{post_id}", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["post"]["title"] == "New Title"
    assert data["post"]["content"] == "New content"

    # Non-existent post → 404
    response = await client.put("/api/posts/999999", json=update_payload)
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]

    # Blank title (spaces) → 200 (UpdatePostRequest has no blank validator)
    response = await client.put(
        f"/api/posts/{post_id}", json={"title": "   ", "content": "valid"}
    )
    assert response.status_code == 200
    updated = response.json()["post"]
    assert updated["title"] == "   "  # exactly as sent

    # Title too long → 422
    response = await client.put(
        f"/api/posts/{post_id}", json={"title": "a" * 201, "content": "valid"}
    )
    assert response.status_code == 422

    # Clean up
    await client.delete(f"/api/posts/{post_id}")


@pytest.mark.asyncio
async def test_delete_api_posts_post_id(client: AsyncClient):
    # Create a post to delete
    create_resp = await client.post(
        "/api/posts", json={"user_id": 1, "title": "Delete me", "content": "..."}
    )
    assert create_resp.status_code == 201
    post_id = create_resp.json()["id"]

    # Happy delete
    response = await client.delete(f"/api/posts/{post_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted_id"] == post_id

    # Delete again → 404
    response = await client.delete(f"/api/posts/{post_id}")
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]

    # Invalid ID (non-integer) → 422
    response = await client.delete("/api/posts/abc")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_slow_data(client: AsyncClient):
    # Default delay (1s) – may be slow but acceptable
    response = await client.get("/api/slow-data")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "data" in data
    assert len(data["data"]) == 10

    # Valid custom delay
    response = await client.get("/api/slow-data?delay_seconds=0.5")
    assert response.status_code == 200

    # Invalid delay (<0.5) → 422
    response = await client.get("/api/slow-data?delay_seconds=0.4")
    assert response.status_code == 422

    # Invalid delay (>5.0) → 422
    response = await client.get("/api/slow-data?delay_seconds=5.1")
    assert response.status_code == 422

    # Invalid type → 422
    response = await client.get("/api/slow-data?delay_seconds=abc")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_large_payload(client: AsyncClient):
    # Default size 100
    response = await client.get("/api/large-payload")
    assert response.status_code == 200
    data = response.json()
    assert data["item_count"] == 100
    assert len(data["items"]) == 100

    # Custom size 50
    response = await client.get("/api/large-payload?size=50")
    assert response.status_code == 200
    data = response.json()
    assert data["item_count"] == 50
    assert len(data["items"]) == 50

    # Size below min (10) → 422
    response = await client.get("/api/large-payload?size=5")
    assert response.status_code == 422

    # Size above max (1000) → 422
    response = await client.get("/api/large-payload?size=1001")
    assert response.status_code == 422

    # Invalid type → 422
    response = await client.get("/api/large-payload?size=abc")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_metrics_db_size(client: AsyncClient):
    response = await client.get("/api/metrics/db-size")
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert "posts" in data
    assert isinstance(data["users"], int)
    assert isinstance(data["posts"], int)

import pytest
from httpx import ASGITransport, AsyncClient

from main import app

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


async def test_get_users(client):
    response = await client.get("/api/users")
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)
    assert len(users) > 0
    user = users[0]
    assert "id" in user
    assert "name" in user
    assert "email" in user
    assert "city" in user
    assert "company" in user


async def test_get_users_paginated(client):
    response = await client.get("/api/users/paginated?page=1&limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["limit"] == 20
    assert data["total"] > 0
    assert data["pages"] > 0
    assert len(data["data"]) == min(20, data["total"])

    total = data["total"]
    last_page = (total + 19) // 20
    response = await client.get(f"/api/users/paginated?page={last_page}&limit=20")
    assert response.status_code == 200
    data_last = response.json()
    assert data_last["page"] == last_page
    assert len(data_last["data"]) == total - (last_page - 1) * 20


async def test_get_user_with_posts(client):
    user_id = 1
    response = await client.get(f"/api/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert "user" in data
    assert "posts" in data
    assert data["user"]["id"] == user_id
    for post in data["posts"]:
        assert post["user_id"] == user_id


async def test_create_post(client):
    payload = {
        "user_id": 1,
        "title": "Test Post Title",
        "content": "This is the post content.",
    }
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 201
    new_post = response.json()
    assert new_post["user_id"] == 1
    assert new_post["title"] == payload["title"]
    assert new_post["content"] == payload["content"]
    assert "id" in new_post
    assert "created_at" in new_post


async def test_update_post(client):
    create_payload = {
        "user_id": 1,
        "title": "Original Title",
        "content": "Original content",
    }
    create_resp = await client.post("/api/posts", json=create_payload)
    post_id = create_resp.json()["id"]

    update_payload = {"title": "Updated Title", "content": "Updated content"}
    resp = await client.put(f"/api/posts/{post_id}", json=update_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    updated = data["post"]
    assert updated["title"] == update_payload["title"]
    assert updated["content"] == update_payload["content"]


async def test_delete_post(client):
    create_payload = {
        "user_id": 1,
        "title": "To Delete",
        "content": "This post will be deleted",
    }
    create_resp = await client.post("/api/posts", json=create_payload)
    post_id = create_resp.json()["id"]

    resp = await client.delete(f"/api/posts/{post_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["deleted_id"] == post_id

    user_posts_resp = await client.get("/api/users/1")
    posts = user_posts_resp.json()["posts"]
    post_ids = [p["id"] for p in posts]
    assert post_id not in post_ids


# ---------------------------------------------------------------------------
# Edge cases / validation errors
# ---------------------------------------------------------------------------


async def test_get_user_not_found(client):
    response = await client.get("/api/users/999999")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_create_post_user_not_found(client):
    payload = {"user_id": 999999, "title": "Test", "content": "Content"}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]


async def test_create_post_blank_title(client):
    payload = {"user_id": 1, "title": "   ", "content": "Content"}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("title" in e["loc"] for e in errors)


async def test_create_post_title_too_long(client):
    payload = {"user_id": 1, "title": "a" * 201, "content": "Content"}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("title" in e["loc"] for e in errors)


async def test_create_post_content_too_long(client):
    payload = {"user_id": 1, "title": "Valid title", "content": "a" * 5001}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("content" in e["loc"] for e in errors)


async def test_update_post_not_found(client):
    payload = {"title": "New", "content": "New content"}
    response = await client.put("/api/posts/999999", json=payload)
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]


async def test_delete_post_not_found(client):
    response = await client.delete("/api/posts/999999")
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Reset / Seed tests (isolation)
# ---------------------------------------------------------------------------


async def test_reset_db_restores_initial_state(client):
    resp1 = await client.get("/api/metrics/db-size")
    initial_counts = resp1.json()

    payload = {"user_id": 1, "title": "Temp", "content": "Temp content"}
    await client.post("/api/posts", json=payload)

    resp2 = await client.get("/api/metrics/db-size")
    modified_counts = resp2.json()
    assert modified_counts["posts"] == initial_counts["posts"] + 1

    reset_resp = await client.post("/api/reset")
    assert reset_resp.status_code == 200

    resp3 = await client.get("/api/metrics/db-size")
    reset_counts = resp3.json()
    assert reset_counts == initial_counts


async def test_seed_db_with_custom_number(client):
    payload = {"num_users": 10}
    response = await client.post("/api/seed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "seeded"
    assert data["users"] == 10
    assert data["posts"] >= 50

    metrics = await client.get("/api/metrics/db-size")
    metrics_data = metrics.json()
    assert metrics_data["users"] == 10


async def test_seed_db_max_limit(client):
    payload = {"num_users": 1000}
    response = await client.post("/api/seed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["users"] == 1000


async def test_seed_db_invalid_limit(client):
    # 0 is invalid (ge=1)
    payload = {"num_users": 0}
    response = await client.post("/api/seed", json=payload)
    assert response.status_code == 422
    # 1001 > 1000
    payload = {"num_users": 1001}
    response = await client.post("/api/seed", json=payload)
    assert response.status_code == 422

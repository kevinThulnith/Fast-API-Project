import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
async def reset_db(client):
    "Reset users_db/posts_db before each test for isolation."
    response = await client.post("/api/reset")
    assert response.status_code == 200
    return response.json()


@pytest.fixture
async def auth_headers(client: AsyncClient):
    """
    Register a fresh user and log in, returning Authorization headers.
    create_post / update_post / delete_post require auth in main.py, so
    every write test below needs this. Email/username are randomized
    since /api/reset does not clear auth_users_db.
    """
    unique = uuid.uuid4().hex[:12]
    register_payload = {
        "email": f"gen-{unique}@example.com",
        "username": f"genuser{unique}",
        "password": "GenPass1234",
    }
    register_resp = await client.post("/api/auth/register", json=register_payload)
    assert register_resp.status_code == 201

    login_resp = await client.post(
        "/api/auth/login",
        data={
            "username": register_payload["email"],
            "password": register_payload["password"],
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Tests
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

    # Missing 'enabled' query param -> 422
    response = await client.post("/api/error-injection")
    assert response.status_code == 422

    # Invalid boolean string -> 422
    response = await client.post("/api/error-injection?enabled=maybe")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_error_injection_forces_500_on_api_routes(client: AsyncClient):
    """
    Confirms the observability middleware actually short-circuits /api/*
    routes when injection is enabled, and that /api/reset, /api/health,
    and /api/error-injection itself remain exempt (otherwise you could
    never turn it back off).
    """
    try:
        response = await client.post("/api/error-injection?enabled=true")
        assert response.status_code == 200

        forced = await client.get("/api/users")
        assert forced.status_code == 500
        assert "Forced server error" in forced.json()["detail"]

        # Exempt paths still work while injection is active
        health_resp = await client.get("/api/health")
        assert health_resp.status_code == 200

        toggle_resp = await client.post("/api/error-injection?enabled=false")
        assert toggle_resp.status_code == 200
    finally:
        # Always disable, even on assertion failure, so later tests aren't poisoned
        await client.post("/api/error-injection?enabled=false")

    recovered = await client.get("/api/users")
    assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_reset_clears_error_injection(client: AsyncClient):
    "/api/reset explicitly resets _force_error back to False per main.py."
    await client.post("/api/error-injection?enabled=true")
    reset_resp = await client.post("/api/reset")
    assert reset_resp.status_code == 200

    response = await client.get("/api/users")
    assert response.status_code == 200


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

    # Invalid page (<1) -> 422
    response = await client.get("/api/users/paginated?page=0")
    assert response.status_code == 422

    # Limit > 100 -> 422 (API enforces le=100, not capped)
    response = await client.get("/api/users/paginated?limit=200")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_api_users_paginated_last_page_remainder(client: AsyncClient):
    """Last page should hold exactly the remainder, not a full page."""
    seed_resp = await client.post("/api/seed", json={"num_users": 25})
    assert seed_resp.status_code == 200

    response = await client.get("/api/users/paginated?page=2&limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 25
    assert data["pages"] == 2
    assert len(data["data"]) == 5


@pytest.mark.asyncio
async def test_get_api_users_user_id(client: AsyncClient):
    # User with ID 1 exists
    response = await client.get("/api/users/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["id"] == 1
    assert "posts" in data
    assert isinstance(data["posts"], list)

    # Non-existent user -> 404
    response = await client.get("/api/users/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # Invalid ID (non-integer) -> 422
    response = await client.get("/api/users/abc")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_api_posts(client: AsyncClient, auth_headers):
    user_id = 1

    # Happy path
    payload = {"user_id": user_id, "title": "Test Post", "content": "Content"}
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 201
    post = response.json()
    assert post["user_id"] == user_id
    assert post["title"] == payload["title"]
    assert post["content"] == payload["content"]
    assert "id" in post
    assert "created_at" in post

    # User not found -> 404
    payload["user_id"] = 999999
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]

    # Missing title -> 422
    payload["user_id"] = user_id
    del payload["title"]
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 422

    # Blank title -> 422
    payload["title"] = "   "
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 422

    # Title too long (>200) -> 422
    payload["title"] = "a" * 201
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 422

    # Content too long (>5000) -> 422
    payload["title"] = "Valid title"
    payload["content"] = "a" * 5001
    response = await client.post("/api/posts", json=payload, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_api_posts_requires_auth(client: AsyncClient):
    """Without a Bearer token, create_post must 401, not 201."""
    payload = {"user_id": 1, "title": "No Auth", "content": "Should fail"}
    response = await client.post("/api/posts", json=payload)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_post_api_posts_rejects_bad_token(client: AsyncClient):
    payload = {"user_id": 1, "title": "Bad Token", "content": "Should fail"}
    response = await client.post(
        "/api/posts", json=payload, headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_put_api_posts_post_id(client: AsyncClient, auth_headers):
    # Create a post to update
    create_resp = await client.post(
        "/api/posts",
        json={"user_id": 1, "title": "Old", "content": "Old content"},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    post_id = create_resp.json()["id"]

    # Happy update
    update_payload = {"title": "New Title", "content": "New content"}
    response = await client.put(
        f"/api/posts/{post_id}", json=update_payload, headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["post"]["title"] == "New Title"
    assert data["post"]["content"] == "New content"

    # Non-existent post -> 404
    response = await client.put(
        "/api/posts/999999", json=update_payload, headers=auth_headers
    )
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]

    # Blank title (spaces) -> 200 (UpdatePostRequest has no blank validator)
    response = await client.put(
        f"/api/posts/{post_id}",
        json={"title": "   ", "content": "valid"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    updated = response.json()["post"]
    assert updated["title"] == "   "  # exactly as sent

    # Title too long -> 422
    response = await client.put(
        f"/api/posts/{post_id}",
        json={"title": "a" * 201, "content": "valid"},
        headers=auth_headers,
    )
    assert response.status_code == 422

    # Clean up
    await client.delete(f"/api/posts/{post_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_put_api_posts_post_id_requires_auth(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/posts",
        json={"user_id": 1, "title": "Old", "content": "Old content"},
        headers=auth_headers,
    )
    post_id = create_resp.json()["id"]

    response = await client.put(
        f"/api/posts/{post_id}", json={"title": "X", "content": "Y"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_delete_api_posts_post_id(client: AsyncClient, auth_headers):
    # Create a post to delete
    create_resp = await client.post(
        "/api/posts",
        json={"user_id": 1, "title": "Delete me", "content": "..."},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    post_id = create_resp.json()["id"]

    # Happy delete
    response = await client.delete(f"/api/posts/{post_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted_id"] == post_id

    # Delete again -> 404
    response = await client.delete(f"/api/posts/{post_id}", headers=auth_headers)
    assert response.status_code == 404
    assert "Post not found" in response.json()["detail"]

    # Invalid ID (non-integer) -> 422
    response = await client.delete("/api/posts/abc", headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_api_posts_post_id_requires_auth(
    client: AsyncClient, auth_headers
):
    create_resp = await client.post(
        "/api/posts",
        json={"user_id": 1, "title": "Delete me", "content": "..."},
        headers=auth_headers,
    )
    post_id = create_resp.json()["id"]

    response = await client.delete(f"/api/posts/{post_id}")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_api_slow_data(client: AsyncClient):
    # Default delay (1s) - may be slow but acceptable
    response = await client.get("/api/slow-data")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "data" in data
    assert len(data["data"]) == 10

    # Valid custom delay
    response = await client.get("/api/slow-data?delay_seconds=0.5")
    assert response.status_code == 200

    # Invalid delay (<0.5) -> 422
    response = await client.get("/api/slow-data?delay_seconds=0.4")
    assert response.status_code == 422

    # Invalid delay (>5.0) -> 422
    response = await client.get("/api/slow-data?delay_seconds=5.1")
    assert response.status_code == 422

    # Invalid type -> 422
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

    # Size below min (10) -> 422
    response = await client.get("/api/large-payload?size=5")
    assert response.status_code == 422

    # Size above max (1000) -> 422
    response = await client.get("/api/large-payload?size=1001")
    assert response.status_code == 422

    # Invalid type -> 422
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


# ---------------------------------------------------------------------------
# Auth: register / login / refresh / me / password / admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    payload = {
        "email": "regtest@example.com",
        "username": "regtestuser",
        "password": "RegPass1234",
    }
    response = await client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == payload["email"]
    assert data["username"] == payload["username"]
    assert data["role"] == "user"
    assert data["is_active"] is True
    assert "hashed_password" not in data
    assert "password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email_rejected(client: AsyncClient):
    payload = {
        "email": "dupe-email@example.com",
        "username": "dupeemailuser1",
        "password": "DupePass1234",
    }
    first = await client.post("/api/auth/register", json=payload)
    assert first.status_code == 201

    payload["username"] = "dupeemailuser2"
    second = await client.post("/api/auth/register", json=payload)
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_register_duplicate_username_case_insensitive_rejected(
    client: AsyncClient,
):
    """Username uniqueness check in main.py is case-insensitive."""
    payload = {
        "email": "caseone@example.com",
        "username": "CaseUser",
        "password": "CasePass1234",
    }
    first = await client.post("/api/auth/register", json=payload)
    assert first.status_code == 201

    payload2 = {
        "email": "casetwo@example.com",
        "username": "caseuser",  # same username, different case
        "password": "CasePass1234",
    }
    second = await client.post("/api/auth/register", json=payload2)
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_register_weak_password_rejected(client: AsyncClient):
    # No uppercase, no digit
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "weak@example.com",
            "username": "weakuser",
            "password": "weakpassword",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_password_too_short_rejected(client: AsyncClient):
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "short@example.com",
            "username": "shortuser",
            "password": "Ab1",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_with_email(client: AsyncClient):
    payload = {
        "email": "loginemail@example.com",
        "username": "loginemailuser",
        "password": "LoginPass1234",
    }
    await client.post("/api/auth/register", json=payload)

    response = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_with_username(client: AsyncClient):
    "main.py's login accepts either email or username in the username field."
    payload = {
        "email": "loginuname@example.com",
        "username": "loginunameuser",
        "password": "LoginPass1234",
    }
    await client.post("/api/auth/register", json=payload)

    response = await client.post(
        "/api/auth/login",
        data={"username": payload["username"], "password": payload["password"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password_rejected(client: AsyncClient):
    payload = {
        "email": "wrongpass@example.com",
        "username": "wrongpassuser",
        "password": "RightPass1234",
    }
    await client.post("/api/auth/register", json=payload)

    response = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": "WrongPass1234"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user_rejected(client: AsyncClient):
    response = await client.post(
        "/api/auth/login",
        data={"username": "nobody@example.com", "password": "Whatever1234"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me_returns_current_user(client: AsyncClient, auth_headers):
    response = await client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"].startswith("gen-")
    assert data["role"] == "user"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_get_me_requires_auth(client: AsyncClient):
    response = await client.get("/api/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_rotates_and_revokes_old(client: AsyncClient):
    payload = {
        "email": "refreshflow@example.com",
        "username": "refreshflowuser",
        "password": "RefreshPass1234",
    }
    await client.post("/api/auth/register", json=payload)
    login_resp = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    refresh_token = login_resp.json()["refresh_token"]

    response = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] != refresh_token

    # Old refresh token is revoked (rotation) — reuse must fail
    reuse_resp = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert reuse_resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_garbage_token_rejected(client: AsyncClient):
    response = await client.post(
        "/api/auth/refresh", json={"refresh_token": "not.a.valid.jwt"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_refresh_token(client: AsyncClient):
    payload = {
        "email": "logoutuser@example.com",
        "username": "logoutuser",
        "password": "LogoutPass1234",
    }
    await client.post("/api/auth/register", json=payload)
    login_resp = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    tokens = login_resp.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    logout_resp = await client.post(
        "/api/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers,
    )
    assert logout_resp.status_code == 200
    assert logout_resp.json()["success"] is True

    # Revoked refresh token can no longer mint new tokens
    refresh_resp = await client.post(
        "/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert refresh_resp.status_code == 401


@pytest.mark.asyncio
async def test_change_password_then_old_password_fails(client: AsyncClient):
    payload = {
        "email": "changepw@example.com",
        "username": "changepwuser",
        "password": "OldPass1234",
    }
    await client.post("/api/auth/register", json=payload)
    login_resp = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    change_resp = await client.put(
        "/api/auth/me/password",
        json={"current_password": "OldPass1234", "new_password": "NewPass5678"},
        headers=headers,
    )
    assert change_resp.status_code == 200

    # Old password no longer works
    old_login = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": "OldPass1234"},
    )
    assert old_login.status_code == 401

    # New password works
    new_login = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": "NewPass5678"},
    )
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_change_password_wrong_current_rejected(
    client: AsyncClient, auth_headers
):
    response = await client.put(
        "/api/auth/me/password",
        json={"current_password": "WrongCurrent123", "new_password": "WontMatter1"},
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_non_admin_cannot_list_users(client: AsyncClient, auth_headers):
    response = await client.get("/api/auth/users", headers=auth_headers)
    assert response.status_code == 403


@pytest.fixture
async def admin_headers(client: AsyncClient):
    """
    Register a user, then directly flip its role to 'admin' in the in-memory
    store. Note: ASGITransport does not invoke FastAPI's lifespan, so the
    bootstrap admin in main.py's lifespan never actually exists under this
    test setup — we can't log in as admin@example.com here.
    """
    from main import _auth_users_by_id  # local import to avoid module-load order issues

    unique = uuid.uuid4().hex[:12]
    register_payload = {
        "email": f"admin-{unique}@example.com",
        "username": f"adminuser{unique}",
        "password": "AdminPass1234",
    }
    register_resp = await client.post("/api/auth/register", json=register_payload)
    assert register_resp.status_code == 201
    new_admin_id = register_resp.json()["id"]
    _auth_users_by_id[new_admin_id].role = "admin"

    login_resp = await client.post(
        "/api/auth/login",
        data={
            "username": register_payload["email"],
            "password": register_payload["password"],
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_admin_can_list_and_manage_users(
    client: AsyncClient, auth_headers, admin_headers
):

    list_resp = await client.get("/api/auth/users", headers=admin_headers)
    assert list_resp.status_code == 200
    users = list_resp.json()
    assert any(u["role"] == "admin" for u in users)

    # Promote the auth_headers test user to admin, then demote back
    me_resp = await client.get("/api/auth/me", headers=auth_headers)
    target_id = me_resp.json()["id"]

    promote_resp = await client.patch(
        f"/api/auth/users/{target_id}/role?role=admin", headers=admin_headers
    )
    assert promote_resp.status_code == 200
    assert promote_resp.json()["role"] == "admin"

    deactivate_resp = await client.patch(
        f"/api/auth/users/{target_id}/deactivate", headers=admin_headers
    )
    assert deactivate_resp.status_code == 200
    assert deactivate_resp.json()["is_active"] is False

    # Deactivated user's existing access token is now rejected
    deactivated_check = await client.get("/api/auth/me", headers=auth_headers)
    assert deactivated_check.status_code == 403


@pytest.mark.asyncio
async def test_admin_route_requires_admin_role_not_just_auth(client: AsyncClient):
    "Confirms require_admin actually checks role, using a non-admin token."
    payload = {
        "email": "notadmin@example.com",
        "username": "notadminuser",
        "password": "NotAdmin1234",
    }
    await client.post("/api/auth/register", json=payload)
    login_resp = await client.post(
        "/api/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    response = await client.patch(
        "/api/auth/users/some-fake-id/role?role=admin", headers=headers
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_role_update_rejects_invalid_role_value(
    client: AsyncClient, admin_headers
):
    response = await client.patch(
        "/api/auth/users/some-id/role?role=superuser", headers=admin_headers
    )
    assert response.status_code == 422

import asyncio
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from faker import Faker
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator

# ---------------------------------------------------------------------------
# Logging setup — structured so LLM agents can parse request/response info
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("llm_ctf_api")

# ---------------------------------------------------------------------------
# Config via environment variables — keeps CI lightweight
# ---------------------------------------------------------------------------
NUM_USERS: int = int(os.getenv("NUM_USERS", "500"))
BASE_DELAY: float = float(os.getenv("BASE_DELAY", "0.05"))

fake = Faker()

# ---------------------------------------------------------------------------
# In-memory DB + reset helpers (test isolation)
# ---------------------------------------------------------------------------
users_db: List["User"] = []
posts_db: List["Post"] = []
_startup_time: float = 0.0

# Flag that lets test agents inject forced errors to exercise self-healing
_force_error: bool = False


def _seed_db(num_users: int = NUM_USERS) -> None:
    """Populate users_db and posts_db with deterministic fake data."""
    global users_db, posts_db
    Faker.seed(42)  # deterministic so LLM-generated assertions are stable
    random.seed(42)
    users_db = []
    posts_db = []
    for i in range(1, num_users + 1):
        users_db.append(
            User(
                id=i,
                name=fake.name(),
                email=fake.email(),
                city=fake.city(),
                company=fake.company(),
            )
        )
        for _ in range(random.randint(5, 10)):
            posts_db.append(
                Post(
                    id=len(posts_db) + 1,
                    user_id=i,
                    title=fake.sentence(nb_words=6),
                    content=fake.paragraph(nb_sentences=3),
                    created_at=fake.iso8601(),
                )
            )
    logger.info("DB seeded: %d users, %d posts", len(users_db), len(posts_db))


# ---------------------------------------------------------------------------
# Pydantic models with field constraints (gives TestGenAgent meaningful edges)
# ---------------------------------------------------------------------------
class Post(BaseModel):
    id: int
    user_id: int
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)
    created_at: str


class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    city: str = Field(..., min_length=1, max_length=100)
    company: str = Field(..., min_length=1, max_length=200)


class CreatePostRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be blank")
        return v


class UpdatePostRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)


class SeedRequest(BaseModel):
    num_users: int = Field(
        10, ge=1, le=1000, description="How many users to seed (default 10)"
    )


# ---------------------------------------------------------------------------
# App lifespan — seed DB on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    _seed_db()
    yield


app = FastAPI(
    title="LLM-CTF Target API",
    description=(
        "FastAPI backend designed as a continuous-testing target for the "
        "LLM-CTF research framework. Includes test hooks for state reset, "
        "deterministic seeding, error injection, and structured observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request ID + structured request logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    # Error injection: lets TestSelectAgent verify self-healing behaviour
    if (
        _force_error
        and request.url.path.startswith("/api/")
        and request.url.path
        not in ("/api/reset", "/api/health", "/api/error-injection")
    ):
        logger.warning(
            "Forced error active | request_id=%s path=%s", request_id, request.url.path
        )
        return Response(
            content='{"detail":"Forced server error (error injection active)"}',
            status_code=500,
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )

    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Health endpoint — CI pipelines poll this before running test suites
# ---------------------------------------------------------------------------
@app.get(
    "/api/health",
    tags=["observability"],
    summary="Health check with uptime and DB size",
)
async def health():
    """
    Returns service status, uptime in seconds, and current DB record counts.
    Used by CI/CD pipelines to gate test execution until the service is ready.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "db": {"users": len(users_db), "posts": len(posts_db)},
    }


# ---------------------------------------------------------------------------
# Test hooks — state management for isolated, repeatable test runs
# ---------------------------------------------------------------------------
@app.post(
    "/api/reset", tags=["test-hooks"], summary="Reset DB to a fresh deterministic state"
)
async def reset_db():
    """
    Wipes and re-seeds the in-memory database using a fixed random seed.
    Call this between test runs to guarantee isolation.
    The data produced is identical on every call (Faker seed=42).
    """
    global _force_error
    _force_error = False
    _seed_db()
    return {"status": "reset", "users": len(users_db), "posts": len(posts_db)}


@app.post(
    "/api/seed",
    tags=["test-hooks"],
    summary="Seed a custom number of users for targeted test scenarios",
)
async def seed_db(body: SeedRequest):
    """
    Seeds the database with a specific number of users (1–1000).
    Useful when a test needs to assert exact pagination boundaries or
    verify behaviour at known dataset sizes.
    """
    _seed_db(num_users=body.num_users)
    return {"status": "seeded", "users": len(users_db), "posts": len(posts_db)}


@app.post(
    "/api/error-injection",
    tags=["test-hooks"],
    summary="Toggle forced 500 errors on all API endpoints",
)
async def set_error_injection(enabled: bool = Query(...)):
    """
    When enabled=true every non-hook API call returns HTTP 500.
    Use this to verify that your TestSelectAgent's self-healing logic
    correctly detects and responds to a degraded service.
    """
    global _force_error
    _force_error = enabled
    logger.warning("Error injection set to %s", enabled)
    return {"error_injection": _force_error}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
@app.get(
    "/api/users",
    response_model=List[User],
    tags=["users"],
    summary="Return all users (no pagination)",
)
async def get_users():
    """Returns the full user list. Simulates a 50 ms network delay."""
    await asyncio.sleep(BASE_DELAY)
    return users_db


@app.get("/api/users/paginated", tags=["users"], summary="Paginated user listing")
async def get_users_paginated(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(20, ge=1, le=100, description="Records per page"),
):
    """
    Returns a page of users with total count metadata.
    Useful for testing pagination edge cases (last page, oversized limit, etc.).
    """
    await asyncio.sleep(BASE_DELAY)
    start = (page - 1) * limit
    end = start + limit
    return {
        "page": page,
        "limit": limit,
        "total": len(users_db),
        "pages": -(-len(users_db) // limit),  # ceiling division
        "data": users_db[start:end],
    }


@app.get(
    "/api/users/{user_id}", tags=["users"], summary="Get a single user with their posts"
)
async def get_user_with_posts(user_id: int):
    """
    Returns a user object together with all their posts.
    Raises 404 if the user_id does not exist.
    """
    await asyncio.sleep(BASE_DELAY)
    user = next((u for u in users_db if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    posts = [p for p in posts_db if p.user_id == user_id]
    return {"user": user, "posts": posts}


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------
@app.post("/api/posts", status_code=201, tags=["posts"], summary="Create a new post")
async def create_post(post: CreatePostRequest):
    """
    Creates a post linked to an existing user.
    Raises 404 if user_id does not exist.
    Validates title (non-blank, ≤200 chars) and content (≤5000 chars).
    """
    await asyncio.sleep(BASE_DELAY * 2)  # simulate DB write latency
    if not any(u.id == post.user_id for u in users_db):
        raise HTTPException(status_code=404, detail="User not found")
    new_post = Post(
        id=len(posts_db) + 1,
        user_id=post.user_id,
        title=post.title,
        content=post.content,
        created_at=fake.iso8601(),
    )
    posts_db.append(new_post)
    return new_post


@app.put(
    "/api/posts/{post_id}", tags=["posts"], summary="Update a post's title and content"
)
async def update_post(post_id: int, body: UpdatePostRequest):
    """
    Updates title and content of an existing post.
    Raises 404 if post_id does not exist.
    """
    await asyncio.sleep(BASE_DELAY * 2)
    for p in posts_db:
        if p.id == post_id:
            p.title = body.title
            p.content = body.content
            return {"success": True, "post": p}
    raise HTTPException(status_code=404, detail="Post not found")


@app.delete("/api/posts/{post_id}", tags=["posts"], summary="Delete a post by ID")
async def delete_post(post_id: int):
    """Deletes the post with the given ID. Raises 404 if not found."""
    global posts_db
    await asyncio.sleep(BASE_DELAY)
    new_posts = [p for p in posts_db if p.id != post_id]
    if len(new_posts) == len(posts_db):
        raise HTTPException(status_code=404, detail="Post not found")
    posts_db = new_posts
    return {"success": True, "deleted_id": post_id}


# ---------------------------------------------------------------------------
# Performance / stress endpoints (for flakiness and load testing)
# ---------------------------------------------------------------------------
@app.get(
    "/api/slow-data",
    tags=["performance"],
    summary="Endpoint with configurable response delay",
)
async def get_slow_data(
    delay_seconds: float = Query(
        1.0, ge=0.5, le=5.0, description="Simulated delay in seconds (0.5–5.0)"
    ),
):
    """
    Sleeps for delay_seconds before responding.
    Used by TestSelectAgent to detect and flag slow/flaky tests that depend
    on timing rather than behaviour.
    """
    await asyncio.sleep(delay_seconds)
    return {
        "message": f"Delayed response after {delay_seconds}s",
        "data": [fake.sentence() for _ in range(10)],
    }


@app.get(
    "/api/large-payload",
    tags=["performance"],
    summary="Return a large payload to stress render/parse time",
)
async def get_large_payload(
    size: int = Query(
        100, ge=10, le=1000, description="Number of paragraphs to return (10–1000)"
    ),
):
    """
    Returns `size` fake paragraphs.
    Useful for testing how the agent handles large response bodies and whether
    generated tests avoid brittle length assertions.
    """
    await asyncio.sleep(BASE_DELAY * 2)
    return {"item_count": size, "items": [fake.paragraph() for _ in range(size)]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get(
    "/api/metrics/db-size",
    tags=["observability"],
    summary="Current record counts in the in-memory DB",
)
async def get_db_metrics():
    """Returns user and post counts. Useful for verifying reset/seed operations."""
    return {"users": len(users_db), "posts": len(posts_db)}

import asyncio
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from faker import Faker
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("llm_ctf_api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_USERS: int = int(os.getenv("NUM_USERS", "500"))
BASE_DELAY: float = float(os.getenv("BASE_DELAY", "0.05"))

# JWT / Auth config — override via env in production
SECRET_KEY: str = os.getenv(
    "SECRET_KEY", "change-me-in-production-use-a-long-random-string"
)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

fake = Faker()

# ---------------------------------------------------------------------------
# Password hashing + OAuth2 scheme
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# In-memory "DB"
# ---------------------------------------------------------------------------
users_db: List["User"] = []
posts_db: List["Post"] = []

auth_users_db: Dict[str, "AuthUser"] = {}  # keyed by email
_auth_users_by_id: Dict[str, "AuthUser"] = {}  # keyed by id — O(1) lookup
_auth_users_by_username: Dict[str, "AuthUser"] = {}  # keyed by username (lowercased)
refresh_tokens_db: Dict[str, "RefreshTokenRecord"] = {}  # keyed by token string

_users_by_id: Dict[int, "User"] = {}
_posts_by_user_id: Dict[int, List["Post"]] = {}
_posts_by_id: Dict[int, "Post"] = {}
_startup_time: float = 0.0
_force_error: bool = False
_db_lock = asyncio.Lock()
_auth_lock = asyncio.Lock()  # guards auth_users_db / its indexes


def _rebuild_indexes() -> None:
    global _users_by_id, _posts_by_user_id, _posts_by_id
    _users_by_id = {u.id: u for u in users_db}
    _posts_by_id = {p.id: p for p in posts_db}
    _posts_by_user_id = {}
    for p in posts_db:
        _posts_by_user_id.setdefault(p.user_id, []).append(p)


def _seed_db_sync(num_users: int = NUM_USERS) -> None:
    global users_db, posts_db
    Faker.seed(42)
    random.seed(42)
    new_users: List["User"] = []
    new_posts: List["Post"] = []
    for i in range(1, num_users + 1):
        new_users.append(
            User(
                id=i,
                name=fake.name(),
                email=fake.email(),
                city=fake.city(),
                company=fake.company(),
            )
        )
        for _ in range(random.randint(5, 10)):
            new_posts.append(
                Post(
                    id=len(new_posts) + 1,
                    user_id=i,
                    title=fake.sentence(nb_words=6),
                    content=fake.paragraph(nb_sentences=3),
                    created_at=fake.iso8601(),
                )
            )
    users_db = new_users
    posts_db = new_posts
    _rebuild_indexes()
    logger.info("DB seeded: %d users, %d posts", len(users_db), len(posts_db))


async def _seed_db(num_users: int = NUM_USERS) -> None:
    async with _db_lock:
        await asyncio.to_thread(_seed_db_sync, num_users)


def _register_auth_user(user: "AuthUser") -> None:
    "Insert into all three auth indexes. Caller must hold _auth_lock."
    auth_users_db[user.email] = user
    _auth_users_by_id[user.id] = user
    _auth_users_by_username[user.username.lower()] = user


def _find_auth_user_by_login(identifier: str) -> Optional["AuthUser"]:
    "Look up by email first, then by username (case-insensitive)."
    user = auth_users_db.get(identifier)
    if user:
        return user
    return _auth_users_by_username.get(identifier.lower())


# ---------------------------------------------------------------------------
# Pydantic models — data
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
    num_users: int = Field(10, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Pydantic models — auth
# ---------------------------------------------------------------------------
class AuthUser(BaseModel):
    """Stored in auth_users_db — never returned directly to callers."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    hashed_password: str
    role: str = Field(default="user")  # "user" | "admin"
    is_active: bool = True
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("password must contain at least one digit")
        return v


class UserPublic(BaseModel):
    """Safe subset of AuthUser — what we return to callers."""

    id: str
    email: EmailStr
    username: str
    role: str
    is_active: bool
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshTokenRecord(BaseModel):
    token: str
    user_id: str
    expires_at: datetime
    revoked: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("new_password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("new_password must contain at least one digit")
        return v


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    payload["iat"] = datetime.now(timezone.utc)
    payload["jti"] = str(uuid.uuid4())  # unique token ID — useful for revocation
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user: AuthUser) -> str:
    return _create_token(
        {"sub": user.id, "email": user.email, "role": user.role, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user: AuthUser) -> str:
    token = _create_token(
        {"sub": user.id, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    # Persist so we can revoke it later
    record = RefreshTokenRecord(
        token=token,
        user_id=user.id,
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    refresh_tokens_db[token] = record
    return token


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthUser:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")
    user_id: str = payload.get("sub", "")
    user = _auth_users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


async def require_admin(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    await _seed_db()
    # Seed one default admin so the API is immediately usable after startup
    _bootstrap_admin()
    yield


def _bootstrap_admin() -> None:
    """Create a default admin account if none exists — dev convenience only."""
    admin_email = "admin@example.com"
    if admin_email not in auth_users_db:
        admin = AuthUser(
            email=admin_email,
            username="admin",
            hashed_password=hash_password("Admin1234"),
            role="admin",
        )
        _register_auth_user(admin)
        logger.info("Bootstrap admin created: %s / Admin1234", admin_email)


app = FastAPI(
    title="Fast API with Auth",
    description=(
        "FastAPI backend with JWT authentication. Includes access tokens, "
        "refresh tokens, role-based access control, and test hooks."
    ),
    version="2.1.0",
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
# Observability middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    if (
        _force_error
        and request.url.path.startswith("/api/")
        and request.url.path
        not in ("/api/reset", "/api/health", "/api/error-injection")
    ):
        logger.warning(
            "Forced error | request_id=%s path=%s", request_id, request.url.path
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


# ===========================================================================
# AUTH ROUTES  /api/auth/*
# ===========================================================================


@app.post(
    "/api/auth/register",
    response_model=UserPublic,
    status_code=201,
    tags=["auth"],
    summary="Register a new account",
)
async def register(body: RegisterRequest):
    """
    Creates a new user account.
    - Email must be unique.
    - Username must be unique (case-insensitive).
    - Password must be ≥8 chars, contain an uppercase letter and a digit.
    - New accounts default to the `user` role.
    """
    async with _auth_lock:
        if body.email in auth_users_db:
            raise HTTPException(status_code=409, detail="Email already registered")
        if body.username.lower() in _auth_users_by_username:
            raise HTTPException(status_code=409, detail="Username already taken")

        user = AuthUser(
            email=body.email,
            username=body.username,
            hashed_password=hash_password(body.password),
        )
        _register_auth_user(user)
        logger.info("New user registered: %s (%s)", user.email, user.id)
        return UserPublic(**user.model_dump())


@app.post(
    "/api/auth/login",
    response_model=TokenResponse,
    tags=["auth"],
    summary="Login — returns access + refresh tokens",
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Accepts `username` (this field may contain either the account's email
    OR its username) and `password` as form fields (OAuth2 standard).
    Returns a short-lived access token (default 30 min) and a long-lived
    refresh token (default 7 days).
    """
    user = _find_auth_user_by_login(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user)
    logger.info("Login: %s", user.email)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.post(
    "/api/auth/refresh",
    response_model=TokenResponse,
    tags=["auth"],
    summary="Exchange a refresh token for a new token pair",
)
async def refresh_tokens(body: RefreshRequest):
    """
    Validates the refresh token, revokes it (rotation), and issues a new pair.
    Raises 401 if the token is expired, revoked, or invalid.
    """
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    record = refresh_tokens_db.get(body.refresh_token)
    if not record or record.revoked:
        raise HTTPException(status_code=401, detail="Refresh token revoked or unknown")
    if record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = _auth_users_by_id.get(record.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    # Rotate — old token is revoked immediately
    record.revoked = True
    access_token = create_access_token(user)
    new_refresh_token = create_refresh_token(user)
    logger.info("Token rotated for user: %s", user.email)
    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@app.post(
    "/api/auth/logout",
    tags=["auth"],
    summary="Revoke the current refresh token",
)
async def logout(
    body: RefreshRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Revokes the supplied refresh token so it can no longer be used.
    The access token remains valid until it naturally expires — callers
    should discard it client-side.
    """
    record = refresh_tokens_db.get(body.refresh_token)
    if record and record.user_id == current_user.id:
        record.revoked = True
    logger.info("Logout: %s", current_user.email)
    return {"success": True, "message": "Logged out — refresh token revoked"}


@app.get(
    "/api/auth/me",
    response_model=UserPublic,
    tags=["auth"],
    summary="Return the currently authenticated user",
)
async def get_me(current_user: AuthUser = Depends(get_current_user)):
    """Requires a valid Bearer access token."""
    return UserPublic(**current_user.model_dump())


@app.put(
    "/api/auth/me/password",
    tags=["auth"],
    summary="Change password for the currently authenticated user",
)
async def change_password(
    body: ChangePasswordRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Verifies `current_password` then updates to `new_password`.
    Revokes all existing refresh tokens for the account on success.
    """
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.hashed_password = hash_password(body.new_password)

    # Invalidate all refresh tokens — force re-login on other devices
    for record in refresh_tokens_db.values():
        if record.user_id == current_user.id:
            record.revoked = True

    logger.info("Password changed for: %s", current_user.email)
    return {"success": True, "message": "Password updated — please log in again"}


# ---------------------------------------------------------------------------
# Admin-only auth management
# ---------------------------------------------------------------------------


@app.get(
    "/api/auth/users",
    response_model=List[UserPublic],
    tags=["auth"],
    summary="[Admin] List all registered accounts",
)
async def list_auth_users(_admin: AuthUser = Depends(require_admin)):
    """Returns all registered accounts. Requires admin role."""
    return [UserPublic(**u.model_dump()) for u in auth_users_db.values()]


@app.patch(
    "/api/auth/users/{user_id}/deactivate",
    tags=["auth"],
    summary="[Admin] Deactivate an account",
)
async def deactivate_user(
    user_id: str,
    _admin: AuthUser = Depends(require_admin),
):
    """Deactivates an account. Revokes all its refresh tokens immediately."""
    user = _auth_users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Auth user not found")
    user.is_active = False
    for record in refresh_tokens_db.values():
        if record.user_id == user_id:
            record.revoked = True
    logger.info("Admin deactivated user: %s", user.email)
    return {"success": True, "user_id": user_id, "is_active": False}


@app.patch(
    "/api/auth/users/{user_id}/role",
    tags=["auth"],
    summary="[Admin] Change a user's role",
)
async def set_user_role(
    user_id: str,
    role: str = Query(..., pattern="^(user|admin)$"),
    _admin: AuthUser = Depends(require_admin),
):
    """Sets a user's role to either `user` or `admin`."""
    user = _auth_users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Auth user not found")
    user.role = role
    logger.info("Admin set role=%s for user: %s", role, user.email)
    return {"success": True, "user_id": user_id, "role": role}


# ===========================================================================
# EXISTING ROUTES — protected where appropriate
# ===========================================================================


@app.get("/api/health", tags=["observability"])
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "db": {"users": len(users_db), "posts": len(posts_db)},
    }


@app.post("/api/reset", tags=["test-hooks"])
async def reset_db():
    global _force_error
    _force_error = False
    await _seed_db()
    return {"status": "reset", "users": len(users_db), "posts": len(posts_db)}


@app.post("/api/seed", tags=["test-hooks"])
async def seed_db(body: SeedRequest):
    await _seed_db(num_users=body.num_users)
    return {"status": "seeded", "users": len(users_db), "posts": len(posts_db)}


@app.post("/api/error-injection", tags=["test-hooks"])
async def set_error_injection(enabled: bool = Query(...)):
    global _force_error
    _force_error = enabled
    return {"error_injection": _force_error}


# Users — read is public; destructive ops require auth
@app.get("/api/users", response_model=List[User], tags=["users"])
async def get_users():
    await asyncio.sleep(BASE_DELAY)
    return list(users_db)


@app.get("/api/users/paginated", tags=["users"])
async def get_users_paginated(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    await asyncio.sleep(BASE_DELAY)
    start = (page - 1) * limit
    end = start + limit
    return {
        "page": page,
        "limit": limit,
        "total": len(users_db),
        "pages": -(-len(users_db) // limit),
        "data": users_db[start:end],
    }


@app.get("/api/users/{user_id}", tags=["users"])
async def get_user_with_posts(user_id: int):
    await asyncio.sleep(BASE_DELAY)
    user = _users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    posts = _posts_by_user_id.get(user_id, [])
    return {"user": user, "posts": posts}


# Posts — reads are public; writes require a valid login
@app.post(
    "/api/posts",
    status_code=201,
    tags=["posts"],
    summary="Create a post (auth required)",
)
async def create_post(
    post: CreatePostRequest,
    _current_user: AuthUser = Depends(get_current_user),  # 🔒
):
    await asyncio.sleep(BASE_DELAY * 2)
    if post.user_id not in _users_by_id:
        raise HTTPException(status_code=404, detail="User not found")
    async with _db_lock:
        new_post = Post(
            id=len(posts_db) + 1,
            user_id=post.user_id,
            title=post.title,
            content=post.content,
            created_at=fake.iso8601(),
        )
        posts_db.append(new_post)
        _posts_by_id[new_post.id] = new_post
        _posts_by_user_id.setdefault(new_post.user_id, []).append(new_post)
    return new_post


@app.put(
    "/api/posts/{post_id}",
    tags=["posts"],
    summary="Update a post (auth required)",
)
async def update_post(
    post_id: int,
    body: UpdatePostRequest,
    _current_user: AuthUser = Depends(get_current_user),  # 🔒
):
    await asyncio.sleep(BASE_DELAY * 2)
    async with _db_lock:
        p = _posts_by_id.get(post_id)
        if not p:
            raise HTTPException(status_code=404, detail="Post not found")
        p.title = body.title
        p.content = body.content
        return {"success": True, "post": p}


@app.delete(
    "/api/posts/{post_id}",
    tags=["posts"],
    summary="Delete a post (auth required)",
)
async def delete_post(
    post_id: int,
    _current_user: AuthUser = Depends(get_current_user),  # 🔒
):
    await asyncio.sleep(BASE_DELAY)
    async with _db_lock:
        p = _posts_by_id.pop(post_id, None)
        if not p:
            raise HTTPException(status_code=404, detail="Post not found")
        posts_db.remove(p)
        user_posts = _posts_by_user_id.get(p.user_id)
        if user_posts:
            user_posts.remove(p)
    return {"success": True, "deleted_id": post_id}


@app.get("/api/slow-data", tags=["performance"])
async def get_slow_data(delay_seconds: float = Query(1.0, ge=0.5, le=5.0)):
    await asyncio.sleep(delay_seconds)
    return {
        "message": f"Delayed response after {delay_seconds}s",
        "data": [fake.sentence() for _ in range(10)],
    }


@app.get("/api/large-payload", tags=["performance"])
async def get_large_payload(size: int = Query(100, ge=10, le=1000)):
    await asyncio.sleep(BASE_DELAY * 2)
    return {"item_count": size, "items": [fake.paragraph() for _ in range(size)]}


@app.get("/api/metrics/db-size", tags=["observability"])
async def get_db_metrics():
    return {"users": len(users_db), "posts": len(posts_db)}

"""Authentication router for admin login."""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from src.adapters.inbound.http.rate_limit import create_rate_limit_dependency
from src.config import Settings
from src.domain.ports.user_storage import UserStoragePort

# Throttles online password guessing against the single admin account
login_rate_limit_dependency = create_rate_limit_dependency("login", "rate_limit_login")


class LoginRequest(BaseModel):
    """Login request payload."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""

    message: str


class UserResponse(BaseModel):
    """Current user response."""

    username: str
    is_admin: bool


def _set_auth_cookie(response: Response, settings: Settings, *, value: str, max_age: int) -> None:
    """Set (or, with max_age=0, delete) the auth cookie.

    In production the frontend (vercel.app) and API (fly.dev) are different
    sites, so this is a third-party cookie: it must be SameSite=None; Secure;
    Partitioned (CHIPS) to survive third-party-cookie blocking. Python < 3.14's
    http.cookies cannot serialize the Partitioned attribute, so the header is
    built directly instead of using response.set_cookie().

    In development, a plain Lax cookie works for localhost over http.
    """
    if settings.secure_cookies:
        response.headers.append(
            "set-cookie",
            f"access_token={value}; HttpOnly; Max-Age={max_age}; Path=/; "
            "SameSite=None; Secure; Partitioned",
        )
    else:
        response.set_cookie(
            key="access_token",
            value=value,
            httponly=True,
            samesite="lax",
            max_age=max_age,
        )


async def require_admin(
    request: Request, access_token: str | None = Cookie(default=None)
) -> UserResponse:
    """Dependency to require admin authentication.

    Use with FastAPI's Depends() to protect admin endpoints.

    Configuration is read from the application handling the request rather than
    from a module global: with a global, whichever application was constructed
    last supplied the JWT secret for every application in the process, so
    building a second app silently invalidated the first app's sessions.

    Args:
        request: The incoming request, used to reach this app's settings.
        access_token: JWT token from httpOnly cookie.

    Returns:
        UserResponse with authenticated admin user info.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not admin.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Auth not configured")

    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(
            access_token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        username = payload.get("sub")
        is_admin = payload.get("is_admin", False)

        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")

        if not is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        return UserResponse(username=username, is_admin=is_admin)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token") from None


def create_router(user_storage: UserStoragePort, settings: Settings) -> APIRouter:
    """Create the auth router.

    Args:
        user_storage: The user storage instance.
        settings: Application settings.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post(
        "/login",
        response_model=LoginResponse,
        dependencies=[Depends(login_rate_limit_dependency)],
    )
    async def login(request: LoginRequest, response: Response) -> LoginResponse:
        """Authenticate user and set JWT cookie."""
        user = await user_storage.get_user_by_username(request.username)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not await user_storage.verify_password(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create JWT token
        expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
        token_data = {
            "sub": user.username,
            "exp": expire,
            "is_admin": user.is_admin,
        }
        token = jwt.encode(
            token_data,
            settings.jwt_secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )

        _set_auth_cookie(response, settings, value=token, max_age=settings.jwt_expire_minutes * 60)

        return LoginResponse(message="Login successful")

    @router.post("/logout", response_model=LoginResponse)
    async def logout(response: Response) -> LoginResponse:
        """Clear the JWT cookie.

        Deletion attributes must match the login cookie (including
        Partitioned), or browsers will not remove it.
        """
        _set_auth_cookie(response, settings, value="", max_age=0)
        return LoginResponse(message="Logged out")

    @router.get("/me", response_model=UserResponse)
    async def get_current_user(
        access_token: str | None = Cookie(default=None),
    ) -> UserResponse:
        """Get the current authenticated user."""
        if not access_token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            payload = jwt.decode(
                access_token,
                settings.jwt_secret_key.get_secret_value(),
                algorithms=[settings.jwt_algorithm],
            )
            username = payload.get("sub")
            if not username:
                raise HTTPException(status_code=401, detail="Invalid token")

            user = await user_storage.get_user_by_username(username)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")

            return UserResponse(username=user.username, is_admin=user.is_admin)

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token") from None

    return router

"""Authentication router for admin login."""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel

from src.config import Settings
from src.domain.ports.user_storage import UserStoragePort

# Module-level references for require_admin dependency
_settings: Settings | None = None
_user_storage: UserStoragePort | None = None


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


async def require_admin(access_token: str | None = Cookie(default=None)) -> UserResponse:
    """Dependency to require admin authentication.

    Use with FastAPI's Depends() to protect admin endpoints.

    Args:
        access_token: JWT token from httpOnly cookie.

    Returns:
        UserResponse with authenticated admin user info.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not admin.
    """
    if _settings is None:
        raise HTTPException(status_code=500, detail="Auth not configured")

    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(
            access_token,
            _settings.jwt_secret_key,
            algorithms=[_settings.jwt_algorithm],
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
    global _settings, _user_storage
    _settings = settings
    _user_storage = user_storage

    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/login", response_model=LoginResponse)
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
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )

        # Set httpOnly cookie with environment-dependent secure flag
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=settings.secure_cookies,
            samesite="lax",
            max_age=settings.jwt_expire_minutes * 60,
        )

        return LoginResponse(message="Login successful")

    @router.post("/logout", response_model=LoginResponse)
    async def logout(response: Response) -> LoginResponse:
        """Clear the JWT cookie."""
        response.delete_cookie(key="access_token")
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
                settings.jwt_secret_key,
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

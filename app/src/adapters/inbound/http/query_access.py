"""Capability tokens for reading a stored query.

A query_id identifies a query; it does not authorize reading one. Callers that
submit a question receive a short-lived signed token scoped to that single
query, and the read endpoints accept either such a token or an admin session.
Keeping the two separate means ids stay safe to log, list, and display.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Cookie, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Settings

QUERY_READ_SCOPE = "query:read"

# auto_error=False: a missing Authorization header is not an error here,
# because an admin cookie is an equally valid way to authorize the request.
_bearer_scheme = HTTPBearer(auto_error=False)

# Hoisted out of the signature below: the dependency is a closure rather than a
# decorated route, so ruff's B008 does not recognize it as FastAPI injection.
_BEARER_CREDENTIALS = Depends(_bearer_scheme)
_AUTH_COOKIE = Cookie(default=None)


def issue_query_token(query_id: str, settings: Settings) -> str:
    """Mint a capability token granting read access to one query."""
    expire = datetime.now(UTC) + timedelta(minutes=settings.query_token_expire_minutes)
    return jwt.encode(
        {"qid": query_id, "scope": QUERY_READ_SCOPE, "exp": expire},
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def _token_grants_access(token: str, query_id: str, settings: Settings) -> bool:
    """Check a capability token against the query being requested.

    Raises:
        HTTPException: 401 if the token is expired or not verifiable.
    """
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid access token") from None

    # Scope is checked explicitly: an admin login JWT is signed with the same
    # key, and must not double as a capability for an arbitrary query.
    return claims.get("scope") == QUERY_READ_SCOPE and claims.get("qid") == query_id


def _is_admin_session(access_token: str | None, settings: Settings) -> bool:
    """Check the admin cookie, treating an unusable cookie as 'not admin'.

    A stale or malformed cookie must not turn a request that carries a valid
    capability token into a rejection.
    """
    if not access_token:
        return False
    try:
        claims = jwt.decode(
            access_token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.InvalidTokenError:
        return False
    return bool(claims.get("sub")) and bool(claims.get("is_admin", False))


def create_query_access_dependency(settings: Settings) -> Callable[..., Awaitable[None]]:
    """Build the dependency guarding per-query read endpoints.

    Settings are captured in the closure rather than read from a module global,
    so each application instance carries its own configuration.
    """

    async def require_query_access(
        query_id: str,
        credentials: HTTPAuthorizationCredentials | None = _BEARER_CREDENTIALS,
        access_token: str | None = _AUTH_COOKIE,
    ) -> None:
        if _is_admin_session(access_token, settings):
            return

        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="This query requires the access token returned when it was created",
            )

        if not _token_grants_access(credentials.credentials, query_id, settings):
            raise HTTPException(
                status_code=403,
                detail="Access token is not valid for this query",
            )

    return require_query_access

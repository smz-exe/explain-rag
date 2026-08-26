"""Per-endpoint rate limiting.

Two things this module exists to get right:

1. The limiter key must identify the caller, not the infrastructure. Behind a
   layer-7 proxy every request shares one socket peer, so keying on
   request.client.host turns a per-IP limit into a single global bucket that
   one visitor can exhaust for everybody.
2. Limiter state lives on the application, not in a module global, so tests can
   exercise the limit without leaking counts between test cases.
"""

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request
from limits import RateLimitItem, parse
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter

_LOOPBACK = "127.0.0.1"


def client_key_for(request: Request, client_ip_header: str | None) -> str:
    """Derive the rate-limit identity for a request.

    Args:
        request: The incoming request.
        client_ip_header: Header carrying the real client IP, when the
            deployment sits behind a proxy that overwrites it. None means trust
            only the socket peer — a client-supplied header would otherwise let
            anyone forge a fresh identity per request.

    Returns:
        The key identifying this caller.
    """
    if client_ip_header:
        forwarded = request.headers.get(client_ip_header)
        if forwarded:
            # A proxy may append rather than replace; the client it saw is last.
            return forwarded.split(",")[-1].strip()

    client = getattr(request, "client", None)
    if client is None or not client.host:
        return _LOOPBACK
    return client.host


def create_rate_limiter() -> MovingWindowRateLimiter:
    """Create a limiter with its own in-process storage.

    In-process means the limit is per machine: with several machines running,
    the effective limit multiplies by machine count. Moving to a shared store
    (e.g. Redis) is the fix if this ever runs multi-machine.
    """
    return MovingWindowRateLimiter(MemoryStorage())


def create_rate_limit_dependency(
    scope: str, limit_setting: str
) -> Callable[[Request], Awaitable[None]]:
    """Build a dependency enforcing one named rate limit.

    Args:
        scope: Namespace for the limit, so endpoints do not share a bucket.
        limit_setting: Name of the Settings attribute holding the limit string.

    Returns:
        A FastAPI dependency raising 429 once the limit is exceeded.
    """

    async def enforce_rate_limit(request: Request) -> None:
        settings = getattr(request.app.state, "settings", None)
        if not settings or not settings.rate_limit_enabled:
            return

        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            return

        key = client_key_for(request, settings.client_ip_header)
        rate_limit = parse(getattr(settings, limit_setting))
        if not await limiter.hit(rate_limit, scope, key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(_retry_after_seconds(rate_limit))},
            )

    return enforce_rate_limit


def _retry_after_seconds(rate_limit: RateLimitItem) -> int:
    """Window length of the limit, as a Retry-After hint.

    An upper bound rather than the exact remaining time: with a moving window
    the caller is served again once the oldest hit ages out, which is at worst
    a full window away.
    """
    return int(rate_limit.GRANULARITY.seconds * rate_limit.multiples)

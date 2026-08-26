"""HTTP middleware for security headers and request body size limits."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HSTS_VALUE = "max-age=63072000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add baseline security headers to every response.

    HSTS is only emitted when enabled (production), since it would pin
    browsers to HTTPS on local development hosts.
    """

    def __init__(self, app, enable_hsts: bool = False):
        super().__init__(app)
        self._enable_hsts = enable_hsts

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if self._enable_hsts:
            response.headers["Strict-Transport-Security"] = _HSTS_VALUE
        return response


class BodySizeLimitMiddleware:
    """Reject requests whose body exceeds the configured limit.

    Content-Length is checked first, so an oversized upload is refused without
    reading it. That header alone is not enough: a chunked request carries
    none, and the handler downstream would buffer the whole body into memory
    to parse it — a memory-exhaustion vector on endpoints reachable pre-auth.
    So the body is also counted as it arrives and cut off at the limit.

    Written as raw ASGI rather than BaseHTTPMiddleware because the latter hands
    the downstream app the original receive channel, so wrapping the request
    stream there has no effect on what the handler actually reads.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self._max_bytes
        ):
            await self._reject(scope, receive, send)
            return

        # The limit is exactly how much memory we accept buffering per request.
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                break
            body.extend(message.get("body", b""))
            if len(body) > self._max_bytes:
                await self._reject(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "error": "request_too_large",
                "message": f"Request body exceeds the {self._max_bytes}-byte limit.",
            },
        )
        await response(scope, receive, send)

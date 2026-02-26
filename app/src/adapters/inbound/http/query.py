from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from limits import parse
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter
from pydantic import BaseModel
from slowapi.util import get_remote_address

from src.application.query_service import QueryService
from src.domain.entities.query import QueryRequest, QueryResponse
from src.domain.ports.query_storage import QueryNotFoundError

# Module-level rate limiter storage (shared across requests)
_rate_limit_storage = MemoryStorage()
_rate_limiter = MovingWindowRateLimiter(_rate_limit_storage)


async def rate_limit_dependency(request: Request) -> None:
    """Dependency to enforce rate limiting on the query endpoint.

    This dependency checks if rate limiting is enabled and applies the
    configured rate limit (default: 10 requests/minute per IP).

    Raises:
        HTTPException 429: If the rate limit is exceeded.
    """
    settings = getattr(request.app.state, "settings", None)

    if not settings or not settings.rate_limit_enabled:
        return

    # Get client identifier for rate limiting
    key = get_remote_address(request)
    limit_string = settings.rate_limit_query

    # Parse and check rate limit
    rate_limit = parse(limit_string)
    if not await _rate_limiter.hit(rate_limit, "query", key):
        from slowapi.errors import RateLimitExceeded

        # Raise the slowapi exception for consistent error handling
        raise RateLimitExceeded(None)


class QuerySummary(BaseModel):
    """Summary of a stored query."""

    query_id: str
    question: str
    answer_preview: str
    created_at: str


class QueriesResponse(BaseModel):
    """Response model for listing recent queries."""

    queries: list[QuerySummary]
    total: int


def create_router(query_service: QueryService) -> APIRouter:
    """Create the query router.

    Args:
        query_service: The query service instance.

    Returns:
        Configured APIRouter.

    Note:
        Rate limiting is applied via slowapi middleware configured in main.py.
        The /query POST endpoint is rate limited (default: 10/minute per IP).
    """
    router = APIRouter(prefix="/query", tags=["query"])

    @router.post("", response_model=QueryResponse, dependencies=[Depends(rate_limit_dependency)])
    async def query(query_request: QueryRequest) -> QueryResponse:
        """Submit a question and receive an explained answer.

        The response includes:
        - Generated answer with inline citations [1], [2], etc.
        - Retrieved chunks with relevance scores
        - Faithfulness verification with per-claim breakdown
        - Timing trace for the pipeline

        Rate limited to prevent API abuse (default: 10 requests/minute per IP).
        """
        return await query_service.query(query_request)

    @router.get("/{query_id}/explanation", response_model=QueryResponse)
    async def get_explanation(query_id: str) -> QueryResponse:
        """Retrieve the full explanation for a previous query.

        Args:
            query_id: The UUID returned from POST /query.

        Returns:
            The complete QueryResponse for that query.

        Raises:
            404: If query_id not found.
        """
        try:
            return await query_service.get_query(query_id)
        except QueryNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Query not found: {query_id}",
            ) from None

    @router.get("/list", response_model=QueriesResponse)
    async def list_queries(
        limit: int = Query(default=20, ge=1, le=100, description="Max number of queries to return"),
    ) -> QueriesResponse:
        """List recent queries with summary information.

        Returns:
            List of query summaries with id, question, answer preview, and timestamp.
        """
        queries = await query_service.list_recent_queries(limit=limit)
        return QueriesResponse(
            queries=[
                QuerySummary(
                    query_id=q["query_id"],
                    question=q["question"],
                    answer_preview=q["answer_preview"],
                    created_at=q["created_at"],
                )
                for q in queries
            ],
            total=len(queries),
        )

    @router.get("/{query_id}/export")
    async def export_query(query_id: str) -> Response:
        """Export a query as a Markdown file.

        Args:
            query_id: The UUID of the query to export.

        Returns:
            A downloadable Markdown file containing the query details.

        Raises:
            404: If query_id not found.
        """
        try:
            query_response = await query_service.get_query(query_id)
        except QueryNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Query not found: {query_id}",
            ) from None

        markdown = _format_query_as_markdown(query_response)

        return Response(
            content=markdown,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="query-{query_id[:8]}.md"',
            },
        )

    return router


def _format_query_as_markdown(query: QueryResponse) -> str:
    """Format a QueryResponse as Markdown for export.

    Args:
        query: The query response to format.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        "# Query Export",
        "",
        f"**Query ID:** `{query.query_id}`",
        "",
        "## Question",
        "",
        query.question,
        "",
        "## Answer",
        "",
        query.answer,
        "",
        "## Retrieved Chunks",
        "",
    ]

    for i, chunk in enumerate(query.retrieved_chunks, 1):
        score_info = f"Similarity: {chunk.similarity_score:.3f}"
        if chunk.rerank_score is not None:
            score_info += f", Rerank: {chunk.rerank_score:.3f}"

        lines.extend(
            [
                f"### [{i}] {chunk.paper_title}",
                "",
                f"**Scores:** {score_info}",
                "",
                "```",
                chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Faithfulness",
            "",
            f"**Overall Score:** {query.faithfulness.score:.0%}",
            "",
        ]
    )

    if query.faithfulness.claims:
        lines.append("### Claims")
        lines.append("")
        for claim in query.faithfulness.claims:
            verdict_emoji = {"supported": "+", "unsupported": "-", "partial": "~"}.get(
                claim.verdict, "?"
            )
            lines.append(f"- [{verdict_emoji}] **{claim.verdict.upper()}:** {claim.claim}")
        lines.append("")

    lines.extend(
        [
            "## Performance",
            "",
            f"- Embedding: {query.trace.embedding_time_ms:.0f}ms",
            f"- Retrieval: {query.trace.retrieval_time_ms:.0f}ms",
        ]
    )

    if query.trace.reranking_time_ms is not None:
        lines.append(f"- Reranking: {query.trace.reranking_time_ms:.0f}ms")

    lines.extend(
        [
            f"- Generation: {query.trace.generation_time_ms:.0f}ms",
            f"- Faithfulness: {query.trace.faithfulness_time_ms:.0f}ms",
            f"- **Total: {query.trace.total_time_ms:.0f}ms**",
            "",
            "---",
            "",
            "*Exported from ExplainRAG*",
        ]
    )

    return "\n".join(lines)

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.application.query_service import QueryService
from src.domain.entities.query import QueryRequest, QueryResponse
from src.domain.ports.query_storage import QueryNotFoundError


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
    """
    router = APIRouter(prefix="/query", tags=["query"])

    @router.post("", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Submit a question and receive an explained answer.

        The response includes:
        - Generated answer with inline citations [1], [2], etc.
        - Retrieved chunks with relevance scores
        - Faithfulness verification with per-claim breakdown
        - Timing trace for the pipeline
        """
        return await query_service.query(request)

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

    return router

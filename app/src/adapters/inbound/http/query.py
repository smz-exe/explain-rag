from fastapi import APIRouter, HTTPException

from src.application.query_service import QueryNotFoundError, QueryService
from src.domain.entities.query import QueryRequest, QueryResponse


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

    return router

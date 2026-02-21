from fastapi import APIRouter
from pydantic import BaseModel

from src.domain.ports.vector_store import VectorStorePort


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    papers_count: int
    chunks_count: int


def create_router(vector_store: VectorStorePort) -> APIRouter:
    """Create the health router.

    Args:
        vector_store: The vector store instance.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Check the health of the service."""
        stats = await vector_store.get_stats()

        return HealthResponse(
            status="healthy",
            papers_count=stats.get("paper_count", 0),
            chunks_count=stats.get("chunk_count", 0),
        )

    return router

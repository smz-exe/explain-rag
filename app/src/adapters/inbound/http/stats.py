from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.adapters.inbound.http.auth import require_admin
from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.vector_store import VectorStorePort


class SystemStats(BaseModel):
    """Response model for system statistics."""

    papers_count: int
    chunks_count: int
    queries_count: int
    backend_status: str = "healthy"


def create_router(
    vector_store: VectorStorePort,
    query_storage: QueryStoragePort,
) -> APIRouter:
    """Create the stats router.

    Args:
        vector_store: The vector store instance.
        query_storage: The query storage instance.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(tags=["admin"])

    @router.get("/stats", response_model=SystemStats, dependencies=[Depends(require_admin)])
    async def get_stats() -> SystemStats:
        """Get system statistics for admin dashboard."""
        vector_stats = await vector_store.get_stats()
        queries_count = await query_storage.count()

        return SystemStats(
            papers_count=vector_stats.get("paper_count", 0),
            chunks_count=vector_stats.get("chunk_count", 0),
            queries_count=queries_count,
        )

    return router

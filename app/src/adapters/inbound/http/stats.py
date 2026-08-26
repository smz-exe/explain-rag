from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.adapters.inbound.http.auth import require_admin
from src.application.stats_service import StatsService


class SystemStats(BaseModel):
    """Response model for system statistics."""

    papers_count: int
    chunks_count: int
    queries_count: int
    backend_status: str = "healthy"


def create_router(stats_service: StatsService) -> APIRouter:
    """Create the stats router.

    Args:
        stats_service: The service collecting corpus statistics.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(tags=["admin"])

    @router.get("/stats", response_model=SystemStats, dependencies=[Depends(require_admin)])
    async def get_stats() -> SystemStats:
        """Get system statistics for admin dashboard."""
        stats = await stats_service.collect_stats()

        return SystemStats(
            papers_count=stats.papers_count,
            chunks_count=stats.chunks_count,
            queries_count=stats.queries_count,
        )

    return router

"""HTTP endpoints for paper coordinates and clustering."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.adapters.inbound.http.auth import require_admin
from src.application.coordinates_service import CoordinatesService


class PaperWithCoordinates(BaseModel):
    """Paper with 3D coordinates for visualization."""

    paper_id: str
    arxiv_id: str
    title: str
    coords: tuple[float, float, float] = Field(description="[x, y, z] UMAP coordinates")
    cluster_id: int | None = Field(description="Cluster assignment (None for noise)")
    chunk_count: int


class ClusterInfo(BaseModel):
    """Cluster information."""

    id: int
    label: str
    paper_ids: list[str]


class EmbeddingsResponse(BaseModel):
    """Response model for paper embeddings/coordinates."""

    papers: list[PaperWithCoordinates]
    computed_at: str | None = Field(description="ISO timestamp of last computation")


class ClustersResponse(BaseModel):
    """Response model for clusters."""

    clusters: list[ClusterInfo]
    computed_at: str | None = Field(description="ISO timestamp of last computation")


class RecomputeResponse(BaseModel):
    """Response model for recompute operation."""

    papers_processed: int
    clusters_found: int
    time_ms: float


def create_router(coordinates_service: CoordinatesService) -> APIRouter:
    """Create the coordinates router.

    Args:
        coordinates_service: The coordinates service instance.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/papers", tags=["coordinates"])

    @router.get("/embeddings", response_model=EmbeddingsResponse)
    async def get_embeddings() -> EmbeddingsResponse:
        """Get 3D coordinates for all papers.

        Returns precomputed UMAP projections and cluster assignments
        for visualization. Call POST /admin/papers/recompute-embeddings
        to refresh the data.

        Returns:
            List of papers with coordinates and cluster assignments.
        """
        coords = await coordinates_service.get_paper_coordinates()
        computed_at = coordinates_service.computed_at

        return EmbeddingsResponse(
            papers=[
                PaperWithCoordinates(
                    paper_id=c.paper_id,
                    arxiv_id=c.arxiv_id,
                    title=c.title,
                    coords=c.coords,
                    cluster_id=c.cluster_id,
                    chunk_count=c.chunk_count,
                )
                for c in coords
            ],
            computed_at=computed_at.isoformat() if computed_at else None,
        )

    @router.get("/clusters", response_model=ClustersResponse)
    async def get_clusters() -> ClustersResponse:
        """Get cluster information.

        Returns auto-generated clusters with labels based on paper titles.

        Returns:
            List of clusters with labels and paper IDs.
        """
        clusters = await coordinates_service.get_clusters()
        computed_at = coordinates_service.computed_at

        return ClustersResponse(
            clusters=[
                ClusterInfo(
                    id=c.id,
                    label=c.label,
                    paper_ids=c.paper_ids,
                )
                for c in clusters
            ],
            computed_at=computed_at.isoformat() if computed_at else None,
        )

    return router


def create_admin_router(coordinates_service: CoordinatesService) -> APIRouter:
    """Create the admin coordinates router.

    Args:
        coordinates_service: The coordinates service instance.

    Returns:
        Configured APIRouter for admin operations.
    """
    router = APIRouter(prefix="/admin/papers", tags=["admin"])

    @router.post(
        "/recompute-embeddings",
        response_model=RecomputeResponse,
        dependencies=[Depends(require_admin)],
    )
    async def recompute_embeddings() -> RecomputeResponse:
        """Trigger recomputation of embedding coordinates and clusters.

        This operation:
        1. Retrieves all paper embeddings from the vector store
        2. Runs UMAP dimensionality reduction to project to 3D
        3. Runs HDBSCAN clustering to group similar papers
        4. Generates cluster labels from paper titles

        The results are cached in memory until the next recompute.

        Returns:
            Statistics about the recomputation.

        Raises:
            HTTPException: 500 if recomputation fails.
        """
        try:
            result = await coordinates_service.recompute_all()
            return RecomputeResponse(
                papers_processed=result["papers_processed"],
                clusters_found=result["clusters_found"],
                time_ms=result["time_ms"],
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to recompute embeddings: {e}",
            ) from e

    return router

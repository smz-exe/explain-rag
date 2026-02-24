from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.adapters.inbound.http.auth import require_admin
from src.application.ingestion_service import IngestionService


class IngestRequest(BaseModel):
    """Request model for paper ingestion."""

    arxiv_ids: list[str] | None = Field(default=None, description="List of arXiv IDs to ingest")
    search_query: str | None = Field(
        default=None, description="Search query to find papers to ingest"
    )
    max_results: int = Field(default=5, ge=1, le=20, description="Max papers to ingest from search")


class IngestResultItem(BaseModel):
    """Result for a single paper ingestion."""

    arxiv_id: str
    title: str
    chunk_count: int
    status: str
    error: str | None = None


class IngestResponse(BaseModel):
    """Response model for paper ingestion."""

    ingested: list[IngestResultItem]
    errors: list[IngestResultItem]


def create_router(ingestion_service: IngestionService) -> APIRouter:
    """Create the ingestion router.

    Args:
        ingestion_service: The ingestion service instance.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/ingest", tags=["ingestion"])

    @router.post("", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    async def ingest_papers(request: IngestRequest) -> IngestResponse:
        """Ingest papers from arXiv.

        Provide either `arxiv_ids` for specific papers, or `search_query`
        to search and ingest papers. If both are provided, `arxiv_ids` takes precedence.
        """
        if request.arxiv_ids:
            result = await ingestion_service.ingest_papers(request.arxiv_ids)
        elif request.search_query:
            result = await ingestion_service.search_and_ingest(
                request.search_query, request.max_results
            )
        else:
            return IngestResponse(ingested=[], errors=[])

        return IngestResponse(
            ingested=[
                IngestResultItem(
                    arxiv_id=r.arxiv_id,
                    title=r.title,
                    chunk_count=r.chunk_count,
                    status=r.status,
                    error=r.error,
                )
                for r in result.ingested
            ],
            errors=[
                IngestResultItem(
                    arxiv_id=r.arxiv_id,
                    title=r.title,
                    chunk_count=r.chunk_count,
                    status=r.status,
                    error=r.error,
                )
                for r in result.errors
            ],
        )

    return router

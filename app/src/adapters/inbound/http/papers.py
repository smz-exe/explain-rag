import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.adapters.inbound.http.auth import require_admin
from src.domain.ports.paper_source import PaperSourcePort
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


class PaperInfo(BaseModel):
    """Information about an ingested paper."""

    paper_id: str
    arxiv_id: str
    title: str
    chunk_count: int


class PapersResponse(BaseModel):
    """Response model for listing papers."""

    papers: list[PaperInfo]
    total: int


class DeletePaperResponse(BaseModel):
    """Response model for paper deletion."""

    paper_id: str
    deleted_chunks: int


class PaperSearchResult(BaseModel):
    """Paper search result for preview before ingestion."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str


class PaperSearchResponse(BaseModel):
    """Response model for paper search."""

    papers: list[PaperSearchResult]
    total: int


def create_router(
    vector_store: VectorStorePort,
    paper_source: PaperSourcePort | None = None,
) -> APIRouter:
    """Create the papers router.

    Args:
        vector_store: The vector store instance.
        paper_source: Optional paper source for search functionality.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/papers", tags=["papers"])

    @router.get("", response_model=PapersResponse)
    async def list_papers() -> PapersResponse:
        """List all ingested papers."""
        papers = await vector_store.list_papers()

        return PapersResponse(
            papers=[
                PaperInfo(
                    paper_id=p.paper_id,
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    chunk_count=p.chunk_count,
                )
                for p in papers
            ],
            total=len(papers),
        )

    @router.delete(
        "/{paper_id}", response_model=DeletePaperResponse, dependencies=[Depends(require_admin)]
    )
    async def delete_paper(paper_id: str) -> DeletePaperResponse:
        """Delete a paper and all its chunks.

        Args:
            paper_id: The paper ID to delete.

        Returns:
            Deletion result with count of deleted chunks.

        Raises:
            HTTPException: 404 if paper not found.
        """
        deleted_count = await vector_store.delete_paper(paper_id)

        # None means no such paper; 0 means it existed and had no chunks.
        if deleted_count is None:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found: {paper_id}",
            )

        return DeletePaperResponse(
            paper_id=paper_id,
            deleted_chunks=deleted_count,
        )

    @router.get(
        "/search",
        response_model=PaperSearchResponse,
        dependencies=[Depends(require_admin)],
    )
    async def search_papers(
        query: str = Query(..., min_length=2, description="Search query for arXiv"),
        max_results: int = Query(default=5, ge=1, le=20, description="Maximum results"),
    ) -> PaperSearchResponse:
        """Search arXiv for papers without ingesting them.

        This endpoint allows admins to search arXiv and preview results
        before deciding which papers to ingest.

        Args:
            query: Search query string (min 2 characters).
            max_results: Maximum number of results to return (1-20).

        Returns:
            List of matching papers with metadata.

        Raises:
            502: If the arXiv search fails.
            503: If paper source is not configured.
        """
        if paper_source is None:
            raise HTTPException(
                status_code=503,
                detail="Paper search is not available: paper source not configured",
            )

        try:
            papers = await paper_source.search(query, max_results)
        except Exception as e:
            # Detail stays in the log. Interpolating the exception here bypassed
            # the global handler's production sanitizing and shipped internals
            # (URLs, driver messages, stack context) straight to the client.
            logger.exception(f"arXiv search failed for query: {query!r}")
            raise HTTPException(
                status_code=502,
                detail="arXiv search is temporarily unavailable. Please try again.",
            ) from e

        return PaperSearchResponse(
            papers=[
                PaperSearchResult(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=p.authors,
                    abstract=p.abstract[:500] + "..." if len(p.abstract) > 500 else p.abstract,
                    url=p.url,
                )
                for p in papers
            ],
            total=len(papers),
        )

    return router

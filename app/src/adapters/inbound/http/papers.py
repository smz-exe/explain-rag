from fastapi import APIRouter
from pydantic import BaseModel

from src.domain.ports.vector_store import VectorStorePort


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


def create_router(vector_store: VectorStorePort) -> APIRouter:
    """Create the papers router.

    Args:
        vector_store: The vector store instance.

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
                    paper_id=p["paper_id"],
                    arxiv_id=p.get("arxiv_id", ""),
                    title=p.get("title", ""),
                    chunk_count=p.get("chunk_count", 0),
                )
                for p in papers
            ],
            total=len(papers),
        )

    return router

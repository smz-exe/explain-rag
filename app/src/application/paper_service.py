"""Service for listing, deleting, and searching papers."""

import logging
from dataclasses import dataclass

from src.domain.entities.paper import Paper, StoredPaper
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)

# How much of an abstract a search hit carries. Someone deciding what to ingest
# needs the gist, not the full text.
ABSTRACT_PREVIEW_CHARS = 500


@dataclass(frozen=True)
class PaperPreview:
    """A search hit from the paper source, before any decision to ingest it."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str


class PaperSearchUnavailableError(Exception):
    """Raised when the service has no paper source configured to search."""

    pass


class PaperSearchError(Exception):
    """Raised when the paper source fails to answer a search."""

    pass


class PaperService:
    """Service for the ingested paper corpus and for previewing new papers."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        paper_source: PaperSourcePort | None = None,
    ):
        """Initialize the paper service.

        Args:
            vector_store: Adapter holding the ingested papers.
            paper_source: Optional adapter for searching papers not yet ingested.
        """
        self._vector_store = vector_store
        self._paper_source = paper_source

    async def list_papers(self) -> list[StoredPaper]:
        """List every ingested paper.

        Returns:
            Stored papers with their chunk counts.
        """
        return await self._vector_store.list_papers()

    async def delete_paper(self, paper_id: str) -> int:
        """Delete a paper and all its chunks.

        Args:
            paper_id: The paper ID to delete.

        Returns:
            Number of chunks deleted.

        Raises:
            PaperNotFoundError: If no such paper existed.
        """
        deleted_count = await self._vector_store.delete_paper(paper_id)

        # None means no such paper; 0 means it existed and had no chunks.
        if deleted_count is None:
            raise PaperNotFoundError(paper_id)

        return deleted_count

    async def search_papers(self, query: str, max_results: int) -> list[PaperPreview]:
        """Search the paper source without ingesting anything.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Matching papers with abstracts trimmed to a preview length.

        Raises:
            PaperSearchUnavailableError: If no paper source is configured.
            PaperSearchError: If the paper source fails.
        """
        if self._paper_source is None:
            raise PaperSearchUnavailableError("No paper source is configured")

        try:
            papers = await self._paper_source.search(query, max_results)
        except Exception as e:
            # The detail stays in the log. Carrying it out to the caller ships
            # internals (URLs, driver messages, stack context) to the client.
            logger.exception(f"arXiv search failed for query: {query!r}")
            raise PaperSearchError("Paper search failed") from e

        return [self._to_preview(paper) for paper in papers]

    @staticmethod
    def _to_preview(paper: Paper) -> PaperPreview:
        """Trim a paper down to what a search hit shows.

        Args:
            paper: The paper returned by the source.

        Returns:
            A preview with the abstract truncated to ABSTRACT_PREVIEW_CHARS.
        """
        abstract = paper.abstract
        if len(abstract) > ABSTRACT_PREVIEW_CHARS:
            abstract = abstract[:ABSTRACT_PREVIEW_CHARS] + "..."

        return PaperPreview(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=abstract,
            url=paper.url,
        )

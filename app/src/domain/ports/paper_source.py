from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.paper import Paper


class PaperSourcePort(ABC):
    """Abstract interface for fetching academic papers."""

    @abstractmethod
    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Fetch paper metadata by arXiv ID.

        Args:
            arxiv_id: The arXiv identifier (e.g., "2401.12345").

        Returns:
            Paper entity with metadata.

        Raises:
            PaperNotFoundError: If paper does not exist.
        """
        ...

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """Search for papers by keyword.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of Paper entities matching the query.
        """
        ...

    @abstractmethod
    async def extract_chunks(
        self, paper: Paper, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        """Download PDF, parse text, and split into chunks.

        Args:
            paper: Paper entity with pdf_url.
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            List of Chunk entities extracted from the paper.

        Raises:
            PDFParsingError: If PDF cannot be parsed.
        """
        ...


class PaperNotFoundError(Exception):
    """Raised when a paper cannot be found."""

    pass


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed."""

    pass

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

import arxiv
import fitz  # PyMuPDF
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.entities.chunk import Chunk
from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort, PDFParsingError

logger = logging.getLogger(__name__)


class ArxivPaperSource(PaperSourcePort):
    """Paper source adapter using arXiv API."""

    def __init__(self):
        """Initialize the arXiv client."""
        self._client = arxiv.Client()

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _fetch_arxiv_results(client, search):
        """Fetch arXiv results with retry logic for network errors."""
        return list(client.results(search))

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _download_pdf(result, dirpath, filename):
        """Download PDF with retry logic for network errors."""
        return result.download_pdf(dirpath=dirpath, filename=filename)

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Fetch paper metadata by arXiv ID."""
        # Normalize arXiv ID (remove version suffix if present)
        clean_id = arxiv_id.split("v")[0]
        logger.debug(f"Fetching paper metadata for: {arxiv_id}")

        search = arxiv.Search(id_list=[clean_id])

        try:
            results = await asyncio.to_thread(
                self._fetch_arxiv_results, self._client, search
            )
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            raise PaperNotFoundError(f"Failed to fetch paper {arxiv_id}: {e}") from e

        if not results:
            raise PaperNotFoundError(f"Paper not found: {arxiv_id}")

        result = results[0]

        return Paper(
            id=str(uuid.uuid4()),
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            url=result.entry_id,
            pdf_url=result.pdf_url,
        )

    async def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """Search for papers by keyword."""
        logger.debug(f"Searching arXiv for: {query}")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            results = await asyncio.to_thread(
                self._fetch_arxiv_results, self._client, search
            )
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")
            raise

        papers = []
        for result in results:
            paper = Paper(
                id=str(uuid.uuid4()),
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
            )
            papers.append(paper)

        return papers

    async def extract_chunks(
        self, paper: Paper, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        """Download PDF, parse text, and split into chunks."""
        logger.debug(f"Extracting chunks from paper: {paper.arxiv_id}")

        # Download PDF to temp file
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / f"{paper.arxiv_id}.pdf"

            # Download the PDF with retry
            search = arxiv.Search(id_list=[paper.arxiv_id.split("v")[0]])
            try:
                results = await asyncio.to_thread(
                    self._fetch_arxiv_results, self._client, search
                )
            except Exception as e:
                logger.error(f"Failed to fetch paper for download: {paper.arxiv_id}: {e}")
                raise PDFParsingError(
                    f"Could not download PDF for {paper.arxiv_id}: {e}"
                ) from e

            if not results:
                raise PDFParsingError(f"Could not download PDF for {paper.arxiv_id}")

            try:
                await asyncio.to_thread(
                    self._download_pdf,
                    results[0],
                    temp_dir,
                    f"{paper.arxiv_id}.pdf",
                )
            except Exception as e:
                logger.error(f"PDF download failed for {paper.arxiv_id}: {e}")
                raise PDFParsingError(
                    f"Failed to download PDF for {paper.arxiv_id}: {e}"
                ) from e

            # Parse PDF and extract text
            text = await self._extract_text_from_pdf(pdf_path)

            if not text.strip():
                raise PDFParsingError(f"No text extracted from PDF: {paper.arxiv_id}")

            # Split into chunks
            chunks = self._split_text(text, paper.id, chunk_size, chunk_overlap)

            return chunks

    async def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF."""

        def extract():
            doc = None
            try:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page_num, page in enumerate(doc):
                    try:
                        text_parts.append(page.get_text())
                    except Exception as e:
                        # Log but continue with other pages
                        logger.warning(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                return "\n".join(text_parts)
            except Exception as e:
                raise PDFParsingError(
                    f"Failed to open or parse PDF {pdf_path}: {e}"
                ) from e
            finally:
                if doc:
                    doc.close()

        return await asyncio.to_thread(extract)

    def _split_text(
        self, text: str, paper_id: str, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        """Split text into overlapping chunks."""
        # Clean up the text
        text = self._clean_text(text)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_sep = text.rfind(sep, start + chunk_size // 2, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    section=None,
                    metadata={
                        "char_start": start,
                        "char_end": end,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position, accounting for overlap
            start = end - chunk_overlap
            if start >= len(text) or start < 0:
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Replace multiple newlines with double newline
        import re

        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        # Remove hyphenation at line breaks
        text = re.sub(r"-\n", "", text)

        return text.strip()

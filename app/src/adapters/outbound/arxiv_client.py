import asyncio
import tempfile
import uuid
from pathlib import Path

import arxiv
import fitz  # PyMuPDF

from src.domain.entities.chunk import Chunk
from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort, PDFParsingError


class ArxivPaperSource(PaperSourcePort):
    """Paper source adapter using arXiv API."""

    def __init__(self):
        """Initialize the arXiv client."""
        self._client = arxiv.Client()

    async def fetch_by_id(self, arxiv_id: str) -> Paper:
        """Fetch paper metadata by arXiv ID."""
        # Normalize arXiv ID (remove version suffix if present)
        clean_id = arxiv_id.split("v")[0]

        search = arxiv.Search(id_list=[clean_id])

        results = await asyncio.to_thread(list, self._client.results(search))

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
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = await asyncio.to_thread(list, self._client.results(search))

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
        # Download PDF to temp file
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / f"{paper.arxiv_id}.pdf"

            # Download the PDF
            search = arxiv.Search(id_list=[paper.arxiv_id.split("v")[0]])
            results = await asyncio.to_thread(list, self._client.results(search))

            if not results:
                raise PDFParsingError(f"Could not download PDF for {paper.arxiv_id}")

            await asyncio.to_thread(
                results[0].download_pdf, dirpath=temp_dir, filename=f"{paper.arxiv_id}.pdf"
            )

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
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)

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

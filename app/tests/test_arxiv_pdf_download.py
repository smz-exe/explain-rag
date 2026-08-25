"""Tests for the arXiv PDF download path (arxiv >= 4.0 removed Result.download_pdf)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.domain.entities.paper import Paper
from src.domain.ports.paper_source import PDFParsingError


def _make_paper(pdf_url: str) -> Paper:
    return Paper(
        id="00000000-0000-0000-0000-000000000000",
        arxiv_id="2005.11401v3",
        title="Test Paper",
        authors=["Author"],
        abstract="Abstract",
        url="https://arxiv.org/abs/2005.11401v3",
        pdf_url=pdf_url,
    )


class TestDownloadPdf:
    def test_writes_response_content_to_path(self, tmp_path: Path):
        pdf_path = tmp_path / "paper.pdf"
        response = MagicMock()
        response.content = b"%PDF-1.5 fake"
        client = MagicMock()
        client.__enter__.return_value = client
        client.get.return_value = response

        with patch("src.adapters.outbound.arxiv_client.httpx.Client", return_value=client):
            ArxivPaperSource._download_pdf("https://arxiv.org/pdf/2005.11401v3", pdf_path)

        assert pdf_path.read_bytes() == b"%PDF-1.5 fake"
        response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_chunks_rejects_missing_pdf_url(self):
        source = ArxivPaperSource()
        with pytest.raises(PDFParsingError, match="No PDF URL"):
            await source.extract_chunks(_make_paper(pdf_url=""), chunk_size=1000, chunk_overlap=200)

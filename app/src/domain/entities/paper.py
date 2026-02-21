from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Represents an academic paper from arXiv."""

    id: str = Field(description="Internal UUID")
    arxiv_id: str = Field(description="arXiv identifier (e.g., '2401.12345')")
    title: str
    authors: list[str]
    abstract: str
    url: str = Field(description="arXiv URL")
    pdf_url: str = Field(description="Direct PDF URL")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunk_count: int = Field(default=0, description="Number of chunks generated from this paper")

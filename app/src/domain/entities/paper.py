from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


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


class StoredPaper(BaseModel):
    """A paper as the vector store holds it, together with its chunk count.

    Exists so VectorStorePort.list_papers() does not hand callers a dict whose
    keys are whatever the adapter's SELECT happened to alias — a shape the
    test double and the real adapter had no way to agree on.
    """

    model_config = ConfigDict(frozen=True)

    paper_id: str = Field(description="Store-assigned paper UUID")
    arxiv_id: str = Field(default="", description="arXiv identifier")
    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    abstract: str = Field(default="", description="Paper abstract")
    url: str = Field(default="", description="arXiv abstract page")
    pdf_url: str = Field(default="", description="arXiv PDF link")
    ingested_at: str | None = Field(default=None, description="ISO 8601 ingestion time")
    chunk_count: int = Field(default=0, description="Chunks stored for this paper")


class StoreStats(BaseModel):
    """Aggregate counts describing the vector store's contents."""

    model_config = ConfigDict(frozen=True)

    chunk_count: int = Field(description="Total chunks across all papers")
    paper_count: int = Field(description="Total papers")

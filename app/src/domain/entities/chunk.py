from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a chunk of text extracted from a paper."""

    id: str = Field(description="Internal UUID")
    paper_id: str = Field(description="Reference to parent Paper ID")
    content: str = Field(description="Raw text content of the chunk")
    chunk_index: int = Field(description="Position within the paper (0-indexed)")
    section: str | None = Field(default=None, description="Section title if detectable")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

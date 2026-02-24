from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(description="Natural language question")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    paper_ids: list[str] | None = Field(
        default=None, description="Optional: scope query to specific papers"
    )
    enable_reranking: bool = Field(default=False, description="Enable cross-encoder reranking")


class Citation(BaseModel):
    """Maps a claim in the answer to source chunks."""

    claim: str = Field(description="The specific claim/sentence in the answer")
    chunk_ids: list[str] = Field(description="Source chunk IDs supporting this claim")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with scoring metadata."""

    chunk_id: str = Field(description="Chunk UUID")
    paper_id: str = Field(description="Parent paper UUID")
    paper_title: str = Field(description="Title of the source paper")
    content: str = Field(description="Chunk text content")
    similarity_score: float = Field(description="Cosine similarity score")
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder score if reranking enabled"
    )
    original_rank: int = Field(
        description="Rank before reranking (1-indexed), same as rank if reranking disabled"
    )
    rank: int = Field(description="Final rank after retrieval/reranking (1-indexed)")


class GenerationResult(BaseModel):
    """Result from LLM generation with citations."""

    answer: str = Field(description="Generated answer with inline citation markers [1], [2], etc.")
    citations: list[Citation] = Field(description="Citation objects mapping claims to chunks")
    raw_response: str | None = Field(default=None, description="Raw LLM response for debugging")


class QueryResponse(BaseModel):
    """Complete response for a query including all explainability data."""

    query_id: str = Field(description="UUID for retrieving explanation later")
    question: str = Field(description="Original question")
    answer: str = Field(description="Generated answer with inline citations")
    citations: list[Citation] = Field(description="Citation mappings")
    retrieved_chunks: list[RetrievedChunk] = Field(description="Retrieved chunks with scores")
    faithfulness: FaithfulnessResult = Field(description="Faithfulness verification result")
    trace: ExplanationTrace = Field(description="Timing breakdown")

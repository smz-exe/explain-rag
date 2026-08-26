from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult


class QueryRequest(BaseModel):
    """Request model for query endpoint.

    The bounds are not cosmetic: every question is embedded, sent to the LLM,
    and persisted, so an unbounded field is a cost-amplification vector on an
    endpoint anyone can reach.
    """

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language question",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    paper_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Optional: scope query to specific papers",
    )
    enable_reranking: bool = Field(default=False, description="Enable cross-encoder reranking")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only questions, which reach the LLM as empty."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


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
    faithfulness: FaithfulnessResult | None = Field(
        default=None,
        description="Faithfulness verification result (None while verification is pending)",
    )
    faithfulness_status: Literal["completed", "pending", "failed"] = Field(
        default="completed",
        description="Verification lifecycle; 'pending' when verification runs after the answer",
    )
    trace: ExplanationTrace = Field(description="Timing breakdown")

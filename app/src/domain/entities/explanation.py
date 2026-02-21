from pydantic import BaseModel, Field


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str = Field(description="The extracted claim from the answer")
    verdict: str = Field(description="'supported' | 'unsupported' | 'partial'")
    evidence_chunk_ids: list[str] = Field(description="Chunk IDs that support/refute this claim")
    reasoning: str = Field(description="LLM's explanation for the verdict")


class FaithfulnessResult(BaseModel):
    """Overall faithfulness assessment of an answer."""

    score: float = Field(ge=0.0, le=1.0, description="Overall faithfulness score")
    claims: list[ClaimVerification] = Field(description="Per-claim verification results")


class ExplanationTrace(BaseModel):
    """Timing breakdown for the query pipeline."""

    embedding_time_ms: float = Field(description="Time to embed query")
    retrieval_time_ms: float = Field(description="Time to search vector store")
    reranking_time_ms: float | None = Field(
        default=None, description="Time for reranking (if enabled)"
    )
    generation_time_ms: float = Field(description="Time for LLM generation")
    faithfulness_time_ms: float = Field(description="Time for faithfulness verification")
    total_time_ms: float = Field(description="Total pipeline time")

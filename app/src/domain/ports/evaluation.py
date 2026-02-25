"""Abstract interface for RAG evaluation."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """RAGAS evaluation metrics for a RAG response."""

    faithfulness: float = Field(
        ge=0.0, le=1.0, description="Factual consistency of answer against context"
    )
    answer_relevancy: float = Field(
        ge=0.0, le=1.0, description="Relevance of answer to the question"
    )
    context_precision: float = Field(
        ge=0.0, le=1.0, description="Precision of retrieved context"
    )
    context_recall: float = Field(
        ge=0.0, le=1.0, description="Recall of retrieved context (requires ground truth)"
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result for a query."""

    query_id: str
    metrics: EvaluationMetrics
    evaluated_at: str
    evaluation_time_ms: float


class EvaluationPort(ABC):
    """Abstract interface for RAG evaluation using RAGAS or similar frameworks."""

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationMetrics:
        """Evaluate a RAG response using standardized metrics.

        Args:
            question: The original question.
            answer: The generated answer.
            contexts: List of retrieved context chunks.
            ground_truth: Optional ground truth answer for context_recall metric.

        Returns:
            EvaluationMetrics with all computed scores.

        Raises:
            EvaluationError: If evaluation fails.
        """
        ...


class EvaluationError(Exception):
    """Raised when evaluation fails."""

    pass

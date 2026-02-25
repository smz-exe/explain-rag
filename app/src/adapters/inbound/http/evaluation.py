"""Evaluation router for RAGAS metrics."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.adapters.inbound.http.auth import require_admin
from src.domain.ports.evaluation import (
    EvaluationError,
    EvaluationMetrics,
    EvaluationPort,
    EvaluationResult,
)
from src.domain.ports.query_storage import QueryStoragePort


class EvaluateRequest(BaseModel):
    """Optional request body for evaluation with ground truth."""

    ground_truth: str | None = None


def create_router(
    evaluation: EvaluationPort,
    query_storage: QueryStoragePort,
) -> APIRouter:
    """Create the evaluation router.

    Args:
        evaluation: The evaluation adapter (RAGAS).
        query_storage: The query storage for retrieving stored queries.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/evaluation", tags=["admin"])

    @router.post(
        "/query/{query_id}",
        response_model=EvaluationResult,
        dependencies=[Depends(require_admin)],
    )
    async def evaluate_query(
        query_id: str,
        request: EvaluateRequest | None = None,
    ) -> EvaluationResult:
        """Evaluate a stored query using RAGAS metrics.

        This endpoint retrieves a previously stored query and runs RAGAS
        evaluation to compute:
        - Faithfulness: How factually consistent is the answer with the context
        - Answer Relevancy: How relevant is the answer to the question
        - Context Precision: How precise is the retrieved context
        - Context Recall: How well the context covers the answer (requires ground_truth)

        Args:
            query_id: The UUID of the stored query to evaluate.
            request: Optional body with ground_truth for context_recall metric.

        Returns:
            EvaluationResult with all computed metrics.

        Raises:
            404: If query not found.
            500: If evaluation fails.
        """
        # Retrieve stored query
        query = await query_storage.get(query_id)
        if query is None:
            raise HTTPException(
                status_code=404,
                detail=f"Query not found: {query_id}",
            )

        # Extract contexts from retrieved chunks
        contexts = [chunk.content for chunk in query.retrieved_chunks]

        if not contexts:
            raise HTTPException(
                status_code=400,
                detail="Query has no retrieved chunks to evaluate",
            )

        # Run evaluation
        start_time = time.perf_counter()

        try:
            metrics: EvaluationMetrics = await evaluation.evaluate(
                question=query.question,
                answer=query.answer,
                contexts=contexts,
                ground_truth=request.ground_truth if request else None,
            )
        except EvaluationError as e:
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        return EvaluationResult(
            query_id=query_id,
            metrics=metrics,
            evaluated_at=datetime.now(UTC).isoformat(),
            evaluation_time_ms=evaluation_time_ms,
        )

    return router

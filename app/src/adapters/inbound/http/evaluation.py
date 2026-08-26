"""Evaluation router for RAG metrics."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.adapters.inbound.http.auth import require_admin
from src.application.evaluation_service import EvaluationService, NoContextToEvaluateError
from src.domain.ports.evaluation import EvaluationError
from src.domain.ports.query_storage import QueryNotFoundError

logger = logging.getLogger(__name__)


class EvaluateRequest(BaseModel):
    """Optional request body for evaluation with ground truth."""

    ground_truth: str | None = None


class EvaluationMetrics(BaseModel):
    """RAG metrics as published over HTTP."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class EvaluationResult(BaseModel):
    """Response model for a query evaluation.

    Deliberately a separate declaration from the identically-named DTO in
    src.domain.ports.evaluation, which is the contract between the service and
    its adapter. Publishing that one made any change to it a silent change to
    the public API. The names match so the OpenAPI schema — and the client
    generated from it — stay stable.
    """

    query_id: str
    metrics: EvaluationMetrics
    evaluated_at: str
    evaluation_time_ms: float


def create_router(evaluation_service: EvaluationService) -> APIRouter:
    """Create the evaluation router.

    Args:
        evaluation_service: The service scoring stored queries.

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
        """Evaluate a stored query.

        Retrieves a previously stored query and computes:
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
            400: If the query retrieved no chunks to evaluate.
            404: If query not found.
            500: If evaluation fails.
        """
        try:
            result = await evaluation_service.evaluate_query(
                query_id,
                ground_truth=request.ground_truth if request else None,
            )
        except QueryNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Query not found: {query_id}",
            ) from None
        except NoContextToEvaluateError:
            raise HTTPException(
                status_code=400,
                detail="Query has no retrieved chunks to evaluate",
            ) from None
        except EvaluationError as e:
            # str(e) carries the judge's raw output (see anthropic_evaluator's
            # "Judge returned no usable claims: ..."), which must not reach a client.
            logger.exception(f"Evaluation failed for query {query_id}")
            raise HTTPException(
                status_code=500,
                detail="Evaluation failed. Please try again.",
            ) from e

        return EvaluationResult(
            query_id=result.query_id,
            metrics=EvaluationMetrics(
                faithfulness=result.metrics.faithfulness,
                answer_relevancy=result.metrics.answer_relevancy,
                context_precision=result.metrics.context_precision,
                context_recall=result.metrics.context_recall,
            ),
            evaluated_at=result.evaluated_at,
            evaluation_time_ms=result.evaluation_time_ms,
        )

    return router

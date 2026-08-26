"""Service for scoring a stored query with RAG evaluation metrics."""

import time
from datetime import UTC, datetime

from src.domain.ports.evaluation import EvaluationMetrics, EvaluationPort, EvaluationResult
from src.domain.ports.query_storage import QueryNotFoundError, QueryStoragePort


class NoContextToEvaluateError(Exception):
    """Raised when a stored query retrieved no chunks to score an answer against."""

    pass


class EvaluationService:
    """Service orchestrating evaluation of a previously stored query."""

    def __init__(
        self,
        evaluation: EvaluationPort,
        query_storage: QueryStoragePort,
    ):
        """Initialize the evaluation service.

        Args:
            evaluation: Adapter computing the metrics.
            query_storage: Adapter holding previously answered queries.
        """
        self._evaluation = evaluation
        self._query_storage = query_storage

    async def evaluate_query(
        self,
        query_id: str,
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a stored query.

        Computes faithfulness, answer relevancy, and context precision against
        the chunks the query actually retrieved. Context recall needs a
        ground_truth answer to compare with.

        Args:
            query_id: The UUID of the stored query to evaluate.
            ground_truth: Optional reference answer for the context_recall metric.

        Returns:
            EvaluationResult with all computed metrics.

        Raises:
            QueryNotFoundError: If no query is stored under that ID.
            NoContextToEvaluateError: If the query retrieved no chunks.
            EvaluationError: If the evaluation adapter fails.
        """
        query = await self._query_storage.get(query_id)
        if query is None:
            raise QueryNotFoundError(query_id)

        contexts = [chunk.content for chunk in query.retrieved_chunks]
        if not contexts:
            raise NoContextToEvaluateError(query_id)

        start_time = time.perf_counter()

        metrics: EvaluationMetrics = await self._evaluation.evaluate(
            question=query.question,
            answer=query.answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        return EvaluationResult(
            query_id=query_id,
            metrics=metrics,
            evaluated_at=datetime.now(UTC).isoformat(),
            evaluation_time_ms=evaluation_time_ms,
        )

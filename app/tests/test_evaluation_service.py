"""Tests for EvaluationService, which owns the scoring pipeline.

The HTTP router used to hold these: fetching the stored query, deriving the
contexts to score against, and refusing a query that retrieved nothing.
"""

import pytest

from src.application.evaluation_service import EvaluationService, NoContextToEvaluateError
from src.domain.entities.explanation import ExplanationTrace
from src.domain.entities.query import Citation, QueryResponse, RetrievedChunk
from src.domain.ports.evaluation import EvaluationError, EvaluationPort
from src.domain.ports.query_storage import QueryNotFoundError
from tests.conftest import MockEvaluationPort, MockQueryStoragePort


def _retrieved_chunk(chunk_id: str, content: str, rank: int) -> RetrievedChunk:
    """Build a retrieved chunk with the fields evaluation reads."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper-001",
        paper_title="Attention Is All You Need",
        content=content,
        similarity_score=0.9,
        original_rank=rank,
        rank=rank,
    )


def _stored_query(query_id: str = "query-001", chunks: list[RetrievedChunk] | None = None):
    """Build a stored query response."""
    return QueryResponse(
        query_id=query_id,
        question="What is self-attention?",
        answer="Self-attention relates positions of a single sequence [1].",
        citations=[
            Citation(claim="Self-attention relates positions.", chunk_ids=["c1"], confidence=0.9)
        ],
        retrieved_chunks=chunks
        if chunks is not None
        else [
            _retrieved_chunk("c1", "Self-attention is an attention mechanism.", 1),
            _retrieved_chunk("c2", "The Transformer uses stacked self-attention.", 2),
        ],
        trace=ExplanationTrace(
            embedding_time_ms=1.0,
            retrieval_time_ms=2.0,
            generation_time_ms=3.0,
            total_time_ms=6.0,
        ),
    )


class FailingEvaluationPort(EvaluationPort):
    """Evaluation adapter that always fails."""

    async def evaluate(self, question, answer, contexts, ground_truth=None):
        """Fail the way a real adapter reports an unusable response."""
        raise EvaluationError("judge returned unparseable output")


async def _storage_with(query: QueryResponse) -> MockQueryStoragePort:
    """Build a storage double already holding one query."""
    storage = MockQueryStoragePort()
    await storage.store(query)
    return storage


class TestEvaluateQuery:
    """Scoring a stored query."""

    @pytest.mark.asyncio
    async def test_returns_metrics_for_a_stored_query(self):
        """A stored query is scored and stamped with the elapsed time."""
        query = _stored_query()
        service = EvaluationService(
            evaluation=MockEvaluationPort(faithfulness=0.85, answer_relevancy=0.9),
            query_storage=await _storage_with(query),
        )

        result = await service.evaluate_query("query-001")

        assert result.query_id == "query-001"
        assert result.metrics.faithfulness == 0.85
        assert result.metrics.answer_relevancy == 0.9
        assert result.evaluation_time_ms >= 0
        assert result.evaluated_at.endswith("+00:00")

    @pytest.mark.asyncio
    async def test_scores_against_the_retrieved_chunks(self):
        """The contexts handed to the judge are the chunks the query retrieved."""
        evaluator = MockEvaluationPort()
        service = EvaluationService(
            evaluation=evaluator,
            query_storage=await _storage_with(_stored_query()),
        )

        await service.evaluate_query("query-001")

        call = evaluator.evaluate_calls[0]
        assert call["question"] == "What is self-attention?"
        assert call["contexts"] == [
            "Self-attention is an attention mechanism.",
            "The Transformer uses stacked self-attention.",
        ]

    @pytest.mark.asyncio
    async def test_ground_truth_reaches_the_evaluator(self):
        """context_recall needs the reference answer, so it must be passed through."""
        evaluator = MockEvaluationPort(context_recall=0.75)
        service = EvaluationService(
            evaluation=evaluator,
            query_storage=await _storage_with(_stored_query()),
        )

        result = await service.evaluate_query("query-001", ground_truth="A reference answer.")

        assert evaluator.evaluate_calls[0]["ground_truth"] == "A reference answer."
        assert result.metrics.context_recall == 0.75

    @pytest.mark.asyncio
    async def test_omitted_ground_truth_is_none(self):
        """Absent ground truth is passed as None, not an empty string."""
        evaluator = MockEvaluationPort()
        service = EvaluationService(
            evaluation=evaluator,
            query_storage=await _storage_with(_stored_query()),
        )

        await service.evaluate_query("query-001")

        assert evaluator.evaluate_calls[0]["ground_truth"] is None


class TestEvaluateQueryFailures:
    """Conditions the caller has to tell apart."""

    @pytest.mark.asyncio
    async def test_unknown_query_raises(self):
        """An ID nothing is stored under is not an empty evaluation."""
        service = EvaluationService(
            evaluation=MockEvaluationPort(),
            query_storage=MockQueryStoragePort(),
        )

        with pytest.raises(QueryNotFoundError):
            await service.evaluate_query("nonexistent-id")

    @pytest.mark.asyncio
    async def test_query_without_chunks_raises(self):
        """Scoring an answer against no context would report a meaningless zero."""
        evaluator = MockEvaluationPort()
        service = EvaluationService(
            evaluation=evaluator,
            query_storage=await _storage_with(_stored_query(chunks=[])),
        )

        with pytest.raises(NoContextToEvaluateError):
            await service.evaluate_query("query-001")

        assert evaluator.evaluate_calls == []

    @pytest.mark.asyncio
    async def test_evaluator_failure_propagates(self):
        """An adapter failure stays an EvaluationError for the caller to map."""
        service = EvaluationService(
            evaluation=FailingEvaluationPort(),
            query_storage=await _storage_with(_stored_query()),
        )

        with pytest.raises(EvaluationError):
            await service.evaluate_query("query-001")

"""Tests for RAGAS evaluation functionality."""

import pytest

from src.domain.ports.evaluation import EvaluationMetrics, EvaluationPort


class MockEvaluationPort(EvaluationPort):
    """Mock evaluation adapter for testing."""

    def __init__(
        self,
        faithfulness: float = 0.85,
        answer_relevancy: float = 0.90,
        context_precision: float = 0.80,
        context_recall: float = 0.75,
    ):
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._context_precision = context_precision
        self._context_recall = context_recall
        self.evaluate_calls: list[dict] = []

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationMetrics:
        """Return mock evaluation metrics."""
        self.evaluate_calls.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
        return EvaluationMetrics(
            faithfulness=self._faithfulness,
            answer_relevancy=self._answer_relevancy,
            context_precision=self._context_precision,
            context_recall=self._context_recall if ground_truth else 0.0,
        )


@pytest.mark.asyncio
async def test_evaluation_requires_auth(client):
    """Test evaluation endpoint requires authentication."""
    response = await client.post("/evaluation/query/test-id")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_evaluation_query_not_found(authenticated_client):
    """Test evaluation returns 404 for unknown query."""
    response = await authenticated_client.post("/evaluation/query/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_evaluation_returns_metrics(authenticated_client):
    """Test evaluation returns proper metrics structure."""
    # First submit a query to create a stored response
    query_response = await authenticated_client.post(
        "/query",
        json={"question": "What is the Transformer architecture?"},
    )
    assert query_response.status_code == 200
    query_id = query_response.json()["query_id"]

    # Now evaluate it
    eval_response = await authenticated_client.post(f"/evaluation/query/{query_id}")

    # If RAGAS fails (e.g., no API key in tests), skip gracefully
    if eval_response.status_code == 500:
        pytest.skip("RAGAS evaluation unavailable in test environment")

    assert eval_response.status_code == 200
    data = eval_response.json()

    # Check structure
    assert "query_id" in data
    assert "metrics" in data
    assert "evaluated_at" in data
    assert "evaluation_time_ms" in data

    # Check metrics
    metrics = data["metrics"]
    assert "faithfulness" in metrics
    assert "answer_relevancy" in metrics
    assert "context_precision" in metrics
    assert "context_recall" in metrics

    # Check values are in valid range
    for key in ["faithfulness", "answer_relevancy", "context_precision"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


@pytest.mark.asyncio
async def test_evaluation_with_ground_truth(authenticated_client):
    """Test evaluation accepts ground truth for context_recall."""
    # First submit a query
    query_response = await authenticated_client.post(
        "/query",
        json={"question": "What is self-attention?"},
    )
    assert query_response.status_code == 200
    query_id = query_response.json()["query_id"]

    # Evaluate with ground truth
    eval_response = await authenticated_client.post(
        f"/evaluation/query/{query_id}",
        json={"ground_truth": "Self-attention is a mechanism that computes attention scores."},
    )

    if eval_response.status_code == 500:
        pytest.skip("RAGAS evaluation unavailable in test environment")

    assert eval_response.status_code == 200


class TestEvaluationMetrics:
    """Test EvaluationMetrics model."""

    def test_metrics_valid_range(self):
        """Test metrics accept values in [0, 1]."""
        metrics = EvaluationMetrics(
            faithfulness=0.5,
            answer_relevancy=0.75,
            context_precision=1.0,
            context_recall=0.0,
        )
        assert metrics.faithfulness == 0.5
        assert metrics.answer_relevancy == 0.75
        assert metrics.context_precision == 1.0
        assert metrics.context_recall == 0.0

    def test_metrics_rejects_invalid_range(self):
        """Test metrics reject values outside [0, 1]."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvaluationMetrics(
                faithfulness=1.5,  # Invalid
                answer_relevancy=0.75,
                context_precision=0.8,
                context_recall=0.0,
            )

        with pytest.raises(ValidationError):
            EvaluationMetrics(
                faithfulness=0.5,
                answer_relevancy=-0.1,  # Invalid
                context_precision=0.8,
                context_recall=0.0,
            )


class TestMockEvaluationPort:
    """Test the mock evaluation adapter."""

    @pytest.mark.asyncio
    async def test_mock_returns_configured_values(self):
        """Test mock returns configured metric values."""
        mock = MockEvaluationPort(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.7,
            context_recall=0.6,
        )

        metrics = await mock.evaluate(
            question="test question",
            answer="test answer",
            contexts=["context 1", "context 2"],
            ground_truth="test ground truth",
        )

        assert metrics.faithfulness == 0.9
        assert metrics.answer_relevancy == 0.85
        assert metrics.context_precision == 0.7
        assert metrics.context_recall == 0.6

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self):
        """Test mock tracks evaluation calls."""
        mock = MockEvaluationPort()

        await mock.evaluate(
            question="Q1",
            answer="A1",
            contexts=["C1"],
        )
        await mock.evaluate(
            question="Q2",
            answer="A2",
            contexts=["C2"],
            ground_truth="GT",
        )

        assert len(mock.evaluate_calls) == 2
        assert mock.evaluate_calls[0]["question"] == "Q1"
        assert mock.evaluate_calls[1]["ground_truth"] == "GT"

    @pytest.mark.asyncio
    async def test_mock_context_recall_requires_ground_truth(self):
        """Test mock returns 0 for context_recall without ground truth."""
        mock = MockEvaluationPort(context_recall=0.8)

        # Without ground truth
        metrics = await mock.evaluate(
            question="Q",
            answer="A",
            contexts=["C"],
        )
        assert metrics.context_recall == 0.0

        # With ground truth
        metrics = await mock.evaluate(
            question="Q",
            answer="A",
            contexts=["C"],
            ground_truth="GT",
        )
        assert metrics.context_recall == 0.8

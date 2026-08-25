"""Tests for deferred faithfulness verification in QueryService."""

import pytest

from src.application.query_service import QueryService
from src.domain.entities.query import QueryRequest
from src.domain.ports.faithfulness import FaithfulnessPort, FaithfulnessVerificationError
from tests.conftest import (
    MockEmbeddingPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockVectorStorePort,
)


class FailingFaithfulness(FaithfulnessPort):
    """Faithfulness port that always fails, simulating an API outage."""

    async def verify(self, answer, chunks):
        raise FaithfulnessVerificationError("verifier down")


def make_service(sample_chunks, faithfulness=None):
    storage = MockQueryStoragePort()
    service = QueryService(
        embedding=MockEmbeddingPort(),
        vector_store=MockVectorStorePort(chunks=sample_chunks),
        llm=MockLLMPort(),
        faithfulness=faithfulness or MockFaithfulnessPort(),
        query_storage=storage,
    )
    return service, storage


class TestDeferredVerification:
    async def test_deferred_query_returns_pending(self, sample_chunks):
        service, storage = make_service(sample_chunks)

        response = await service.query(
            QueryRequest(question="What is self-attention?"), defer_verification=True
        )

        assert response.faithfulness is None
        assert response.faithfulness_status == "pending"
        assert response.trace.faithfulness_time_ms is None
        assert response.answer  # the answer itself is complete

        stored = await storage.get(response.query_id)
        assert stored is not None
        assert stored.faithfulness_status == "pending"

    async def test_complete_verification_updates_stored_response(self, sample_chunks):
        service, storage = make_service(sample_chunks)
        response = await service.query(
            QueryRequest(question="What is self-attention?"), defer_verification=True
        )

        await service.complete_verification(response)

        stored = await storage.get(response.query_id)
        assert stored.faithfulness_status == "completed"
        assert stored.faithfulness is not None
        assert stored.faithfulness.score == pytest.approx(0.9)
        assert stored.trace.faithfulness_time_ms is not None
        # The original response object is not mutated
        assert response.faithfulness is None
        assert response.faithfulness_status == "pending"

    async def test_verification_failure_marks_failed_without_raising(self, sample_chunks):
        service, storage = make_service(sample_chunks, faithfulness=FailingFaithfulness())
        response = await service.query(
            QueryRequest(question="What is self-attention?"), defer_verification=True
        )

        await service.complete_verification(response)  # must not raise

        stored = await storage.get(response.query_id)
        assert stored.faithfulness_status == "failed"
        assert stored.faithfulness is None

    async def test_synchronous_query_is_unchanged(self, sample_chunks):
        service, storage = make_service(sample_chunks)

        response = await service.query(QueryRequest(question="What is self-attention?"))

        assert response.faithfulness_status == "completed"
        assert response.faithfulness is not None
        assert response.trace.faithfulness_time_ms is not None


class TestDeferredVerificationRoutes:
    """Route-level behavior: pending response, then poll to completed.

    httpx's ASGITransport runs BackgroundTasks before the call returns, so
    polling immediately after POST deterministically sees the final state.
    """

    @pytest.fixture
    async def deferred_client(self, monkeypatch, sample_chunks):
        monkeypatch.setenv("DEFERRED_VERIFICATION", "true")
        from httpx import ASGITransport, AsyncClient

        from src.main import create_app
        from tests.conftest import (
            MockClusteringPort,
            MockCoordinatesStoragePort,
            MockDimensionalityReductionPort,
            MockEvaluationPort,
            MockRerankerPort,
        )

        app = create_app(
            embedding=MockEmbeddingPort(),
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            llm=MockLLMPort(),
            faithfulness=MockFaithfulnessPort(),
            reranker=MockRerankerPort(),
            evaluator=MockEvaluationPort(),
            query_storage=MockQueryStoragePort(),
            coordinates_storage=MockCoordinatesStoragePort(),
            dim_reducer=MockDimensionalityReductionPort(),
            clusterer=MockClusteringPort(),
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_post_returns_pending_then_poll_completes(self, deferred_client):
        response = await deferred_client.post("/query", json={"question": "What is attention?"})

        assert response.status_code == 200
        body = response.json()
        assert body["faithfulness_status"] == "pending"
        assert body["faithfulness"] is None
        assert body["answer"]

        poll = await deferred_client.get(f"/query/{body['query_id']}/faithfulness")
        assert poll.status_code == 200
        poll_body = poll.json()
        assert poll_body["status"] == "completed"
        assert poll_body["faithfulness"]["score"] == pytest.approx(0.9)
        assert poll_body["faithfulness_time_ms"] is not None

    async def test_poll_unknown_query_returns_404(self, deferred_client):
        poll = await deferred_client.get("/query/no-such-id/faithfulness")
        assert poll.status_code == 404

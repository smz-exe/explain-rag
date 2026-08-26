"""Tests for reconciling verifications abandoned by a dead process.

Deferred verification runs as an in-process background task with no durable
queue. If the process is suspended or redeployed mid-verification, the stored
query keeps saying "pending" forever — the record actively lies about work that
will never finish. Startup reconciliation closes that state out.
"""

import pytest

from src.application.query_service import QueryService
from src.domain.entities.explanation import ExplanationTrace
from src.domain.entities.query import QueryResponse
from tests.conftest import (
    MockEmbeddingPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockVectorStorePort,
)


def make_response(query_id: str, status: str) -> QueryResponse:
    return QueryResponse(
        query_id=query_id,
        question="What is self-attention?",
        answer="An attention mechanism.",
        citations=[],
        retrieved_chunks=[],
        faithfulness=None,
        faithfulness_status=status,
        trace=ExplanationTrace(
            embedding_time_ms=1.0,
            retrieval_time_ms=1.0,
            reranking_time_ms=None,
            generation_time_ms=1.0,
            faithfulness_time_ms=None,
            total_time_ms=3.0,
        ),
    )


def make_service(storage: MockQueryStoragePort, sample_chunks) -> QueryService:
    return QueryService(
        embedding=MockEmbeddingPort(),
        vector_store=MockVectorStorePort(chunks=sample_chunks),
        llm=MockLLMPort(),
        faithfulness=MockFaithfulnessPort(),
        query_storage=storage,
    )


class TestAbandonedVerificationReconciliation:
    @pytest.mark.asyncio
    async def test_stale_pending_queries_are_marked_failed(self, sample_chunks):
        """No process is verifying these any more, so 'pending' is a lie."""
        storage = MockQueryStoragePort()
        await storage.store(make_response("stale-1", "pending"))
        await storage.store(make_response("stale-2", "pending"))
        service = make_service(storage, sample_chunks)

        reconciled = await service.reconcile_abandoned_verifications()

        assert reconciled == 2
        for query_id in ("stale-1", "stale-2"):
            stored = await storage.get(query_id)
            assert stored.faithfulness_status == "failed"

    @pytest.mark.asyncio
    async def test_completed_queries_are_left_alone(self, sample_chunks):
        storage = MockQueryStoragePort()
        await storage.store(make_response("done-1", "completed"))
        service = make_service(storage, sample_chunks)

        reconciled = await service.reconcile_abandoned_verifications()

        assert reconciled == 0
        assert (await storage.get("done-1")).faithfulness_status == "completed"

    @pytest.mark.asyncio
    async def test_already_failed_queries_are_not_recounted(self, sample_chunks):
        storage = MockQueryStoragePort()
        await storage.store(make_response("failed-1", "failed"))
        service = make_service(storage, sample_chunks)

        reconciled = await service.reconcile_abandoned_verifications()

        assert reconciled == 0

    @pytest.mark.asyncio
    async def test_no_storage_configured_is_a_no_op(self, sample_chunks):
        service = QueryService(
            embedding=MockEmbeddingPort(),
            vector_store=MockVectorStorePort(chunks=sample_chunks),
            llm=MockLLMPort(),
            faithfulness=MockFaithfulnessPort(),
        )

        assert await service.reconcile_abandoned_verifications() == 0

    @pytest.mark.asyncio
    async def test_a_query_verified_after_reconciliation_still_wins(self, sample_chunks):
        """Reconciliation must not permanently poison a query that does complete."""
        storage = MockQueryStoragePort()
        response = make_response("racing-1", "pending")
        await storage.store(response)
        service = make_service(storage, sample_chunks)

        await service.reconcile_abandoned_verifications()
        await service.complete_verification(response)

        stored = await storage.get("racing-1")
        assert stored.faithfulness_status == "completed"
        assert stored.faithfulness is not None

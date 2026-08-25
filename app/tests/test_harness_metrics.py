"""Tests for the quality-harness metric computations (pure functions)."""

import pytest

from src.adapters.inbound.cli.harness_metrics import (
    first_hit_rank,
    hit_rate_at,
    mean_rank_displacement,
    mrr,
    percentile,
    record_from_response,
    summarize,
)
from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult
from src.domain.entities.query import Citation, QueryResponse, RetrievedChunk


def make_chunk(rank: int, original_rank: int, paper_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{rank}",
        paper_id=paper_id,
        paper_title="T",
        content="c",
        similarity_score=0.5,
        rerank_score=0.9,
        original_rank=original_rank,
        rank=rank,
    )


def make_response(chunks: list[RetrievedChunk], answer: str = "Answer [1].") -> QueryResponse:
    return QueryResponse(
        query_id="q1",
        question="Q?",
        answer=answer,
        citations=[Citation(claim="Answer", chunk_ids=[chunks[0].chunk_id], confidence=0.9)]
        if chunks
        else [],
        retrieved_chunks=chunks,
        faithfulness=FaithfulnessResult(score=0.8, claims=[]),
        trace=ExplanationTrace(
            embedding_time_ms=10.0,
            retrieval_time_ms=20.0,
            reranking_time_ms=30.0,
            generation_time_ms=100.0,
            faithfulness_time_ms=50.0,
            total_time_ms=210.0,
        ),
    )


class TestPureMetrics:
    def test_first_hit_rank_found(self):
        assert first_hit_rank(["a", "b", "a"], "b") == 2

    def test_first_hit_rank_absent(self):
        assert first_hit_rank(["a", "b"], "z") is None

    def test_mean_rank_displacement(self):
        # (original_rank, final_rank): displacements 2, 0, 2
        assert mean_rank_displacement([(3, 1), (2, 2), (1, 3)]) == pytest.approx(4 / 3)

    def test_mean_rank_displacement_empty(self):
        assert mean_rank_displacement([]) == 0.0

    def test_mrr(self):
        assert mrr([1, 2, None]) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_mrr_empty(self):
        assert mrr([]) == 0.0

    def test_hit_rate_at(self):
        assert hit_rate_at([1, 3, None, 2], k=2) == pytest.approx(0.5)

    def test_percentile_median(self):
        assert percentile([1.0, 3.0, 2.0], 50) == 2.0

    def test_percentile_p90(self):
        values = [float(v) for v in range(1, 11)]
        assert percentile(values, 90) == pytest.approx(9.1)

    def test_percentile_empty(self):
        assert percentile([], 50) == 0.0


class TestRecordFromResponse:
    def test_hit_ranks_for_both_orderings(self):
        # Vector order (original_rank): src at 3; reranked order: src at 1
        chunks = [
            make_chunk(rank=1, original_rank=3, paper_id="src"),
            make_chunk(rank=2, original_rank=1, paper_id="other"),
            make_chunk(rank=3, original_rank=2, paper_id="other"),
        ]
        record = record_from_response(
            {"id": "q1", "paper_id": "src", "arxiv_id": "1234.5", "question": "Q?"},
            make_response(chunks),
        )

        assert record["reranked_hit_rank"] == 1
        assert record["vector_hit_rank"] == 3
        assert record["mean_displacement"] == pytest.approx((2 + 1 + 1) / 3)
        assert record["max_displacement"] == 2
        assert record["timings"]["generation_ms"] == 100.0
        assert record["faithfulness_score"] == 0.8
        assert record["n_citations"] == 1
        assert record["insufficient_context"] is False

    def test_insufficient_context_detected(self):
        chunks = [make_chunk(rank=1, original_rank=1, paper_id="other")]
        response = make_response(
            chunks, answer="I cannot answer this question based on the available context."
        )
        record = record_from_response(
            {"id": "q1", "paper_id": "src", "arxiv_id": "1", "question": "Q?"}, response
        )

        assert record["insufficient_context"] is True
        assert record["vector_hit_rank"] is None


class TestSummarize:
    def test_aggregates(self):
        records = [
            {
                "vector_hit_rank": 1,
                "reranked_hit_rank": 2,
                "mean_displacement": 1.0,
                "max_displacement": 2,
                "insufficient_context": False,
                "faithfulness_score": 1.0,
                "timings": {
                    "embedding_ms": 10.0,
                    "retrieval_ms": 20.0,
                    "reranking_ms": 30.0,
                    "generation_ms": 100.0,
                    "faithfulness_ms": 50.0,
                    "total_ms": 210.0,
                },
            },
            {
                "vector_hit_rank": None,
                "reranked_hit_rank": None,
                "mean_displacement": 0.0,
                "max_displacement": 0,
                "insufficient_context": True,
                "faithfulness_score": 0.0,
                "timings": {
                    "embedding_ms": 20.0,
                    "retrieval_ms": 40.0,
                    "reranking_ms": 50.0,
                    "generation_ms": 200.0,
                    "faithfulness_ms": 70.0,
                    "total_ms": 380.0,
                },
            },
        ]

        summary = summarize(records, top_k=10)

        assert summary["n_questions"] == 2
        assert summary["retrieval"]["hit_rate_top_k"] == 0.5
        assert summary["retrieval"]["vector"]["mrr"] == pytest.approx(0.5)
        assert summary["retrieval"]["reranked"]["mrr"] == pytest.approx(0.25)
        assert summary["retrieval"]["reranked"]["hit_rate_at_1"] == 0.0
        assert summary["insufficient_context_count"] == 1
        assert summary["timings_ms"]["generation"]["median"] == 150.0
        assert summary["signals"]["faithfulness_mean"] == pytest.approx(0.5)

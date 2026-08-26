"""Metric computations for the quality harness.

Everything here is a pure function over harness records so it can be unit
tested without touching the pipeline. Only objectively measurable facts are
computed: retrieval hit ranks against the question's source paper (ground
truth by construction), rank displacement introduced by reranking, and
latency percentiles. LLM-judge outputs (faithfulness) are carried through
under "signals" and are not treated as quality evidence.
"""

from typing import Any

from src.domain.entities.query import QueryResponse

_INSUFFICIENT_MARKER = "cannot answer this question"


def first_hit_rank(ordered_paper_ids: list[str], source_paper_id: str) -> int | None:
    """1-based rank of the first chunk from the source paper, or None."""
    for rank, paper_id in enumerate(ordered_paper_ids, start=1):
        if paper_id == source_paper_id:
            return rank
    return None


def mean_rank_displacement(pairs: list[tuple[int, int]]) -> float:
    """Mean |original_rank - final_rank| over (original, final) pairs."""
    if not pairs:
        return 0.0
    return sum(abs(original - final) for original, final in pairs) / len(pairs)


def mrr(hit_ranks: list[int | None]) -> float:
    """Mean reciprocal rank; misses count as 0."""
    if not hit_ranks:
        return 0.0
    return sum(1.0 / rank for rank in hit_ranks if rank is not None) / len(hit_ranks)


def hit_rate_at(hit_ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose hit rank is within the top k."""
    if not hit_ranks:
        return 0.0
    return sum(1 for rank in hit_ranks if rank is not None and rank <= k) / len(hit_ranks)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0-100) of a list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (pct / 100) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def record_from_response(question: dict[str, Any], response: QueryResponse) -> dict[str, Any]:
    """Build a per-question harness record from a pipeline response.

    The response must come from a reranking-enabled run: each retrieved chunk
    then carries both its vector rank (original_rank) and its final rank, so
    one run yields hit ranks for both orderings.
    """
    chunks = response.retrieved_chunks
    by_vector_order = sorted(chunks, key=lambda c: c.original_rank)
    vector_paper_ids = [c.paper_id for c in by_vector_order]
    reranked_paper_ids = [c.paper_id for c in sorted(chunks, key=lambda c: c.rank)]
    displacement_pairs = [(c.original_rank, c.rank) for c in chunks]

    trace = response.trace
    return {
        "id": question["id"],
        "arxiv_id": question["arxiv_id"],
        "paper_id": question["paper_id"],
        "question": question["question"],
        "insufficient_context": _INSUFFICIENT_MARKER in response.answer.lower(),
        "vector_hit_rank": first_hit_rank(vector_paper_ids, question["paper_id"]),
        "reranked_hit_rank": first_hit_rank(reranked_paper_ids, question["paper_id"]),
        "mean_displacement": mean_rank_displacement(displacement_pairs),
        "max_displacement": max(
            (abs(o - f) for o, f in displacement_pairs),
            default=0,
        ),
        "n_citations": len(response.citations),
        # None, not 0.0: "verification did not run" must stay distinguishable
        # from "every claim was unsupported", or a benchmark run against a
        # deferred backend silently reports a floor of zeros as real scores.
        "n_claims": len(response.faithfulness.claims) if response.faithfulness else None,
        "faithfulness_score": response.faithfulness.score if response.faithfulness else None,
        "timings": {
            "embedding_ms": trace.embedding_time_ms,
            "retrieval_ms": trace.retrieval_time_ms,
            "reranking_ms": trace.reranking_time_ms,
            "generation_ms": trace.generation_time_ms,
            "faithfulness_ms": trace.faithfulness_time_ms,
            "total_ms": trace.total_time_ms,
        },
    }


def _mean_or_none(values: list[float]) -> float | None:
    """Mean of the values, or None when there is nothing to average."""
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate per-question records into the harness summary."""
    vector_ranks = [r["vector_hit_rank"] for r in records]
    reranked_ranks = [r["reranked_hit_rank"] for r in records]

    def ordering_stats(ranks: list[int | None]) -> dict[str, float]:
        return {
            "mrr": mrr(ranks),
            "hit_rate_at_1": hit_rate_at(ranks, 1),
            "hit_rate_at_3": hit_rate_at(ranks, 3),
        }

    def stage_stats(stage_key: str) -> dict[str, float]:
        values = [
            r["timings"][stage_key] for r in records if r["timings"].get(stage_key) is not None
        ]
        return {"median": percentile(values, 50), "p90": percentile(values, 90)}

    return {
        "n_questions": len(records),
        "top_k": top_k,
        "retrieval": {
            # The retrieved set is identical for both orderings (reranking
            # reorders the top-k window rather than expanding it)
            "hit_rate_top_k": hit_rate_at(vector_ranks, top_k),
            "vector": ordering_stats(vector_ranks),
            "reranked": ordering_stats(reranked_ranks),
        },
        "displacement": {
            "mean": (
                sum(r["mean_displacement"] for r in records) / len(records) if records else 0.0
            ),
            "max": max((r["max_displacement"] for r in records), default=0),
        },
        "insufficient_context_count": sum(1 for r in records if r["insufficient_context"]),
        "timings_ms": {
            "embedding": stage_stats("embedding_ms"),
            "retrieval": stage_stats("retrieval_ms"),
            "reranking": stage_stats("reranking_ms"),
            "generation": stage_stats("generation_ms"),
            "faithfulness": stage_stats("faithfulness_ms"),
            "total": stage_stats("total_ms"),
        },
        "signals": {
            # Averaged over the runs that actually have a score; a run with
            # unverified answers reports None rather than a diluted mean.
            "faithfulness_mean": _mean_or_none(
                [r["faithfulness_score"] for r in records if r["faithfulness_score"] is not None]
            ),
            "faithfulness_scored_runs": sum(
                1 for r in records if r["faithfulness_score"] is not None
            ),
        },
    }

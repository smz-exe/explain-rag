"""CLI for running the quality harness over the committed question set.

Drives QueryService in-process (no HTTP, no rate limits, no query-storage
writes) with reranking enabled; each response carries both the vector
ordering (original_rank) and the reranked ordering (rank), so a single run
per question yields hit ranks for both. Results are written as JSON and a
summary is printed as markdown. Methodology and caveats: eval/README.md.

    uv run python -m src.adapters.inbound.cli.quality_harness \
        --env-file .env.production --questions eval/questions.json \
        --out eval/results/2026-08-25.json
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.adapters.inbound.cli.harness_metrics import record_from_response, summarize
from src.adapters.outbound.anthropic_faithfulness import AnthropicFaithfulness
from src.adapters.outbound.anthropic_rag import AnthropicRAG
from src.adapters.outbound.fastembed_embedding import FastEmbedEmbedding
from src.adapters.outbound.fastembed_reranker import FastEmbedReranker
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.application.query_service import QueryService
from src.config import Settings
from src.domain.entities.query import QueryRequest

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the quality harness CLI."""
    parser = argparse.ArgumentParser(
        prog="quality_harness",
        description="Run the fixed question set through the query pipeline and report metrics.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions.json"),
        help="Question set file (default: eval/questions.json)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file with DATABASE_URL etc. (default: .env; use .env.production for prod)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Chunks to retrieve (default: 10)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: eval/results/<UTC date>.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions (smoke runs)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Concurrent pipeline runs (default: 2)"
    )
    return parser


def format_summary_markdown(summary: dict[str, Any], errors: int) -> str:
    """Render the harness summary as a small markdown report."""
    retrieval = summary["retrieval"]
    timings = summary["timings_ms"]
    lines = [
        f"| Questions | {summary['n_questions']} (errors: {errors}) |",
        f"| Source-paper hit rate @ top-{summary['top_k']} | {retrieval['hit_rate_top_k']:.2f} |",
        f"| MRR (vector order) | {retrieval['vector']['mrr']:.3f} |",
        f"| MRR (reranked order) | {retrieval['reranked']['mrr']:.3f} |",
        f"| Hit@1 vector / reranked | {retrieval['vector']['hit_rate_at_1']:.2f} / "
        f"{retrieval['reranked']['hit_rate_at_1']:.2f} |",
        f"| Mean rank displacement (rerank) | {summary['displacement']['mean']:.2f} |",
        f"| Insufficient-context answers | {summary['insufficient_context_count']} |",
        f"| Latency median/p90 total (ms) | {timings['total']['median']:.0f} / "
        f"{timings['total']['p90']:.0f} |",
        f"| ... generation (ms) | {timings['generation']['median']:.0f} / "
        f"{timings['generation']['p90']:.0f} |",
        f"| ... faithfulness (ms) | {timings['faithfulness']['median']:.0f} / "
        f"{timings['faithfulness']['p90']:.0f} |",
        f"| Faithfulness mean (LLM-judge signal, see caveats) | "
        f"{summary['signals']['faithfulness_mean']:.2f} |",
    ]
    return "\n".join(["| Metric | Value |", "|---|---|", *lines])


async def run(args: argparse.Namespace) -> int:
    """Run the harness and write results."""
    # pydantic-settings accepts _env_file at runtime but does not type it
    settings = Settings(_env_file=args.env_file)  # type: ignore[call-arg]

    document = json.loads(args.questions.read_text())
    questions = document["questions"][: args.limit] if args.limit else document["questions"]

    vector_store = PostgresVectorStore(database_url=settings.database_url)
    service = QueryService(
        embedding=FastEmbedEmbedding(model_name=settings.embedding_model),
        vector_store=vector_store,
        llm=AnthropicRAG(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_tokens=settings.claude_max_tokens,
            timeout=settings.claude_timeout,
            max_retries=settings.claude_max_retries,
        ),
        faithfulness=AnthropicFaithfulness(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.claude_timeout,
            max_retries=settings.claude_max_retries,
        ),
        reranker=FastEmbedReranker(model_name=settings.reranker_model),
        query_storage=None,
        default_top_k=args.top_k,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    async def run_question(question: dict[str, Any]) -> None:
        async with semaphore:
            try:
                response = await service.query(
                    QueryRequest(
                        question=question["question"],
                        top_k=args.top_k,
                        enable_reranking=True,
                    )
                )
                record = record_from_response(question, response)
                records.append(record)
                logger.info(
                    f"  {question['arxiv_id']}: hit(vector)={record['vector_hit_rank']} "
                    f"hit(reranked)={record['reranked_hit_rank']} "
                    f"total={record['timings']['total_ms']:.0f}ms"
                )
            except Exception as e:
                logger.error(f"  {question['arxiv_id']}: FAILED — {e}")
                errors.append({"id": question["id"], "error": str(e)})

    try:
        logger.info(f"Running {len(questions)} questions (top_k={args.top_k})")
        await asyncio.gather(*(run_question(q) for q in questions))
    finally:
        await vector_store.close()

    records.sort(key=lambda r: r["arxiv_id"])
    summary = summarize(records, top_k=args.top_k)

    out_path = args.out or Path(f"eval/results/{datetime.now(UTC).date().isoformat()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "run_at": datetime.now(UTC).isoformat(),
                    "question_set_generated_at": document.get("generated_at"),
                    "model": settings.claude_model,
                    "embedding_model": settings.embedding_model,
                    "reranker_model": settings.reranker_model,
                    "top_k": args.top_k,
                },
                "summary": summary,
                "errors": errors,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(f"\nResults written to {out_path}\n")
    print(format_summary_markdown(summary, errors=len(errors)))
    return 1 if errors and not records else 0


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

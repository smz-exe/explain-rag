"""CLI for batch-ingesting arXiv papers into the vector store.

Runs the full ingestion pipeline (arXiv fetch -> PDF chunking -> local
embedding -> pgvector insert) from the command line, so a batch can target
any environment by pointing --env-file at the matching .env file:

    uv run python -m src.adapters.inbound.cli.ingest 2005.11401 1810.04805
    uv run python -m src.adapters.inbound.cli.ingest --file papers.txt --env-file .env.production
    uv run python -m src.adapters.inbound.cli.ingest --search "retrieval augmented generation" --max-results 5
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.adapters.inbound.cli.ingest_plan import parse_ids_text, plan_ingestion
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.adapters.outbound.fastembed_embedding import FastEmbedEmbedding
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.application.ingestion_service import IngestionService
from src.config import Settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Batch-ingest arXiv papers into the ExplainRAG vector store.",
    )
    parser.add_argument("ids", nargs="*", help="arXiv IDs to ingest (e.g. 2005.11401)")
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="File with one arXiv ID per line ('#' starts a comment)",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="arXiv search query; ingests the top --max-results matches",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max papers to ingest from --search (default: 5)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file with DATABASE_URL etc. (default: .env; use .env.production for prod)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the ingestion plan and print it without writing anything",
    )
    return parser


def _collect_requested_ids(args: argparse.Namespace) -> list[str]:
    """Gather arXiv IDs from positional args and --file."""
    ids = list(args.ids)
    if args.file is not None:
        if not args.file.is_file():
            raise FileNotFoundError(f"ID file not found: {args.file}")
        ids.extend(parse_ids_text(args.file.read_text(encoding="utf-8")))
    return ids


async def run(args: argparse.Namespace) -> int:
    """Execute the ingestion batch. Returns a process exit code."""
    # pydantic-settings accepts _env_file at runtime but does not type it
    settings = Settings(_env_file=args.env_file)  # type: ignore[call-arg]

    paper_source = ArxivPaperSource()
    embedding = FastEmbedEmbedding(model_name=settings.embedding_model)
    vector_store = PostgresVectorStore(
        database_url=settings.database_url,
        pool_min_size=1,
        pool_max_size=2,
    )
    service = IngestionService(
        paper_source=paper_source,
        embedding=embedding,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    try:
        requested = _collect_requested_ids(args)

        if args.search:
            print(f"Searching arXiv: {args.search!r} (max {args.max_results})")
            found = await paper_source.search(args.search, args.max_results)
            for paper in found:
                print(f"  found: {paper.arxiv_id}  {paper.title}")
            requested.extend(paper.arxiv_id for paper in found)

        if not requested:
            print("Nothing to ingest: pass arXiv IDs, --file, or --search.")
            return 1

        existing = await vector_store.list_papers()
        plan = plan_ingestion(requested, [p.arxiv_id for p in existing])

        for arxiv_id in plan.skipped:
            print(f"skip (already ingested): {arxiv_id}")

        if not plan.to_ingest:
            print("All requested papers are already ingested.")
            return 0

        print(f"Ingesting {len(plan.to_ingest)} paper(s)...")
        if args.dry_run:
            for arxiv_id in plan.to_ingest:
                print(f"  would ingest: {arxiv_id}")
            return 0

        result = await service.ingest_papers(plan.to_ingest)

        for item in result.ingested:
            print(f"ok:    {item.arxiv_id}  {item.title}  ({item.chunk_count} chunks)")
        for item in result.errors:
            print(f"error: {item.arxiv_id}  {item.error}", file=sys.stderr)

        print(
            f"Done: {len(result.ingested)} ingested, "
            f"{len(result.errors)} failed, {len(plan.skipped)} skipped."
        )
        return 1 if result.errors else 0
    finally:
        await vector_store.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

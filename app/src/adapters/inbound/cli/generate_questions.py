"""CLI for generating the committed evaluation question set.

Deterministic procedure (documented in eval/README.md): one question per
ingested paper, generated from the paper's abstract, papers ordered by
arXiv ID. The source paper_id is recorded as the retrieval ground-truth
label. The output file is committed so harness runs measure a fixed set.

    uv run python -m src.adapters.inbound.cli.generate_questions \
        --env-file .env.production --out eval/questions.json
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.adapters.outbound.anthropic_client import (
    build_client,
    parse_json_response,
    response_text,
)
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.config import Settings

logger = logging.getLogger(__name__)

QUESTION_PROMPT = """You are building an evaluation set for a retrieval system over research papers.

Paper title: {title}

Abstract:
{abstract}

Write ONE question that a researcher might ask which this paper's content should answer.

Rules:
- The question must be answerable from the paper's content
- Paraphrase: do not copy distinctive phrases or rare terms from the abstract
- Do not mention the paper, its title, or its authors in the question
- Keep it to one sentence

Respond with a JSON object: {{"question": "<the question>"}}
Output only the JSON object, no other text:"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the question-generation CLI."""
    parser = argparse.ArgumentParser(
        prog="generate_questions",
        description="Generate the fixed evaluation question set from paper abstracts.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file with DATABASE_URL etc. (default: .env; use .env.production for prod)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/questions.json"),
        help="Output path for the question set (default: eval/questions.json)",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    """Generate one question per paper and write the question set."""
    # pydantic-settings accepts _env_file at runtime but does not type it
    settings = Settings(_env_file=args.env_file)  # type: ignore[call-arg]

    vector_store = PostgresVectorStore(database_url=settings.database_url)
    client = build_client(
        settings.anthropic_api_key.get_secret_value(),
        settings.claude_timeout,
        settings.claude_max_retries,
    )

    try:
        papers = sorted(await vector_store.list_papers(), key=lambda p: p["arxiv_id"])
        logger.info(f"Generating one question for each of {len(papers)} papers")

        questions = []
        for paper in papers:
            prompt = QUESTION_PROMPT.format(title=paper["title"], abstract=paper["abstract"])
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            payload = parse_json_response(response_text(response))
            question = str(payload["question"]).strip()
            questions.append(
                {
                    "id": f"q-{paper['arxiv_id']}",
                    "paper_id": str(paper["paper_id"]),
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "question": question,
                }
            )
            logger.info(f"  {paper['arxiv_id']}: {question}")

        document = {
            "generated_at": datetime.now(UTC).isoformat(),
            "generator_model": settings.claude_model,
            "procedure": (
                "One question per ingested paper, generated from its abstract with the "
                "prompt in generate_questions.py; papers ordered by arXiv ID. The source "
                "paper_id is the retrieval ground-truth label."
            ),
            "questions": questions,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(questions)} questions to {args.out}")
        return 0
    finally:
        await vector_store.close()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

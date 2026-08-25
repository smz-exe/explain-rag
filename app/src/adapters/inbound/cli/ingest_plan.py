"""Pure planning helpers for the ingestion CLI (no I/O)."""

import re
from dataclasses import dataclass

# arXiv IDs look like "2005.11401", "2005.11401v3", or legacy "cs/9901002v1"
_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arXiv ID for duplicate comparison.

    Strips whitespace, lowercases, and removes a trailing version suffix
    so "2005.11401v3" and "2005.11401" compare equal.
    """
    return _VERSION_SUFFIX.sub("", arxiv_id.strip().lower())


def parse_ids_text(text: str) -> list[str]:
    """Parse arXiv IDs from file text: one per line, '#' starts a comment."""
    ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


@dataclass(frozen=True)
class IngestionPlan:
    """Which papers to ingest and which to skip as already present."""

    to_ingest: list[str]
    skipped: list[str]


def plan_ingestion(requested_ids: list[str], existing_arxiv_ids: list[str]) -> IngestionPlan:
    """Split requested IDs into new papers and already-ingested duplicates.

    Re-ingesting an existing arXiv ID would violate the papers.arxiv_id
    unique constraint (paper UUIDs are regenerated per fetch), so papers
    already in the store are skipped. Duplicates within the request itself
    are also dropped, keeping the first occurrence.
    """
    existing = {normalize_arxiv_id(a) for a in existing_arxiv_ids}
    seen: set[str] = set()
    to_ingest: list[str] = []
    skipped: list[str] = []

    for arxiv_id in requested_ids:
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in existing:
            skipped.append(arxiv_id)
        else:
            to_ingest.append(arxiv_id)

    return IngestionPlan(to_ingest=to_ingest, skipped=skipped)

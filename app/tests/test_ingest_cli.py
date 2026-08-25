"""Tests for the batch ingestion CLI."""

from pathlib import Path

import pytest

from src.adapters.inbound.cli.ingest import _collect_requested_ids, build_parser
from src.adapters.inbound.cli.ingest_plan import (
    normalize_arxiv_id,
    parse_ids_text,
    plan_ingestion,
)


class TestNormalizeArxivId:
    def test_strips_version_suffix(self):
        assert normalize_arxiv_id("2005.11401v3") == "2005.11401"

    def test_strips_whitespace_and_lowercases(self):
        assert normalize_arxiv_id("  1706.03762V7 ") == "1706.03762"

    def test_plain_id_unchanged(self):
        assert normalize_arxiv_id("1810.04805") == "1810.04805"

    def test_legacy_id_with_version(self):
        assert normalize_arxiv_id("cs/9901002v1") == "cs/9901002"


class TestParseIdsText:
    def test_one_id_per_line(self):
        assert parse_ids_text("2005.11401\n1810.04805\n") == ["2005.11401", "1810.04805"]

    def test_skips_comments_and_blank_lines(self):
        text = "# corpus\n2005.11401  # RAG\n\n   \n1810.04805\n"
        assert parse_ids_text(text) == ["2005.11401", "1810.04805"]

    def test_empty_text(self):
        assert parse_ids_text("") == []


class TestPlanIngestion:
    def test_skips_already_ingested_ignoring_version(self):
        plan = plan_ingestion(
            ["1706.03762", "2005.11401"],
            existing_arxiv_ids=["1706.03762v7"],
        )
        assert plan.to_ingest == ["2005.11401"]
        assert plan.skipped == ["1706.03762"]

    def test_deduplicates_within_request(self):
        plan = plan_ingestion(
            ["2005.11401", "2005.11401v3", "2005.11401"],
            existing_arxiv_ids=[],
        )
        assert plan.to_ingest == ["2005.11401"]
        assert plan.skipped == []

    def test_drops_empty_ids(self):
        plan = plan_ingestion(["", "  "], existing_arxiv_ids=[])
        assert plan.to_ingest == []
        assert plan.skipped == []

    def test_all_new(self):
        plan = plan_ingestion(["a", "b"], existing_arxiv_ids=[])
        assert plan.to_ingest == ["a", "b"]


class TestCliParsing:
    def test_defaults(self):
        args = build_parser().parse_args(["2005.11401"])
        assert args.ids == ["2005.11401"]
        assert args.env_file == ".env"
        assert args.max_results == 5
        assert not args.dry_run

    def test_collect_ids_from_args_and_file(self, tmp_path: Path):
        id_file = tmp_path / "papers.txt"
        id_file.write_text("# batch\n1810.04805\n2005.11401\n", encoding="utf-8")
        args = build_parser().parse_args(["1706.03762", "--file", str(id_file)])
        assert _collect_requested_ids(args) == ["1706.03762", "1810.04805", "2005.11401"]

    def test_missing_file_raises(self, tmp_path: Path):
        args = build_parser().parse_args(["--file", str(tmp_path / "nope.txt")])
        with pytest.raises(FileNotFoundError):
            _collect_requested_ids(args)

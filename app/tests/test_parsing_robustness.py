"""Tests for code that parses input it does not control.

Chunking runs over arbitrary PDF text with operator-supplied settings, and the
faithfulness and evaluation adapters parse LLM output. All three are places
where "realistic but unexpected" input silently produced wrong results rather
than an error.
"""

import pytest
from pydantic import ValidationError

from src.adapters.outbound.anthropic_evaluator import AnthropicEvaluator
from src.adapters.outbound.anthropic_faithfulness import AnthropicFaithfulness
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.config import Settings
from src.domain.ports.evaluation import EvaluationError


class TestChunking:
    """The chunker must always make forward progress."""

    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap"),
        [(1000, 200), (1000, 499), (100, 49), (40, 19)],
    )
    def test_chunks_cover_the_text_without_repeating_forever(self, chunk_size, chunk_overlap):
        """A sentence-boundary pullback can move the next start backwards.

        `end` may be pulled back to as early as start + chunk_size // 2, and the
        next start is end - chunk_overlap. With a large overlap that is behind
        the current start, so the loop re-emits the same span indefinitely.
        """
        source = ArxivPaperSource()
        # Frequent sentence boundaries, so the pullback path is taken often.
        text = " ".join(f"Sentence number {i} about attention." for i in range(200))

        chunks = source._split_text(text, "paper-1", chunk_size, chunk_overlap)

        assert chunks, "chunker returned nothing"
        # Bounded output proves the loop advanced rather than spinning.
        assert len(chunks) < len(text), "chunk count suggests the loop did not advance"
        starts = [c.metadata["char_start"] for c in chunks]
        assert starts == sorted(starts), "chunk starts moved backwards"
        assert len(set(starts)) == len(starts), "the same span was emitted twice"

    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap"),
        [(1000, 200), (1000, 499), (100, 49), (40, 19)],
    )
    def test_whole_text_is_covered(self, chunk_size, chunk_overlap):
        """No content may be dropped, whatever the configured overlap.

        When the next start lands before the current one the loop breaks on the
        `start < 0` guard, so ingestion stops after a single chunk and the rest
        of the paper is silently discarded — the document looks ingested but
        almost none of it is searchable.
        """
        source = ArxivPaperSource()
        text = " ".join(f"Sentence number {i} about attention." for i in range(200))

        chunks = source._split_text(text, "paper-1", chunk_size, chunk_overlap)

        assert chunks[-1].metadata["char_end"] >= len(source._clean_text(text)), (
            f"only {len(chunks)} chunk(s) produced; the tail of the text was dropped"
        )


class TestChunkingConfiguration:
    """Settings that would break chunking must be refused at startup."""

    @pytest.mark.parametrize("overlap", [500, 600, 999, 1000])
    def test_overlap_at_or_beyond_half_the_chunk_size_is_rejected(self, monkeypatch, overlap):
        monkeypatch.setenv("CHUNK_SIZE", "1000")
        monkeypatch.setenv("CHUNK_OVERLAP", str(overlap))

        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            Settings()

    def test_a_sane_overlap_is_accepted(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "1000")
        monkeypatch.setenv("CHUNK_OVERLAP", "200")

        assert Settings().chunk_overlap == 200


class TestArxivIdNormalization:
    """Version stripping must not corrupt the identifier."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1706.03762", "1706.03762"),
            ("1706.03762v1", "1706.03762"),
            ("1706.03762v12", "1706.03762"),
            # Old-style identifiers whose archive name contains a 'v'.
            ("solv-int/9801001", "solv-int/9801001"),
            ("solv-int/9801001v2", "solv-int/9801001"),
            ("cond-mat/0501001", "cond-mat/0501001"),
        ],
    )
    def test_only_a_trailing_version_suffix_is_removed(self, raw, expected):
        """split('v')[0] truncates at the first 'v' anywhere, not the suffix."""
        assert ArxivPaperSource._normalize_arxiv_id(raw) == expected


class TestClaimDecomposition:
    """The verifier must not treat non-strings as claims."""

    @pytest.mark.asyncio
    async def test_object_shaped_claims_are_unwrapped(self, monkeypatch):
        """A very common LLM deviation: [{"claim": "..."}] instead of ["..."]."""
        adapter = AnthropicFaithfulness(api_key="test")

        async def fake_complete(prompt, max_tokens=None):
            return '[{"claim": "Attention is a mechanism."}, {"claim": "It scales."}]'

        monkeypatch.setattr(adapter, "_complete", fake_complete)

        claims = await adapter._decompose_answer("Attention is a mechanism. It scales.")

        assert claims == ["Attention is a mechanism.", "It scales."]

    @pytest.mark.asyncio
    async def test_unusable_elements_fall_back_to_sentence_split(self, monkeypatch):
        adapter = AnthropicFaithfulness(api_key="test")

        async def fake_complete(prompt, max_tokens=None):
            return "[1, 2, 3]"

        monkeypatch.setattr(adapter, "_complete", fake_complete)

        claims = await adapter._decompose_answer("First claim. Second claim.")

        assert claims == ["First claim.", "Second claim."]


class TestEvaluatorMetrics:
    """Degenerate judge output must not be scored as a perfect result."""

    def test_null_claims_raises_instead_of_crashing(self):
        adapter = AnthropicEvaluator(model="claude-sonnet-5", api_key="test")

        with pytest.raises(EvaluationError):
            adapter._compute_metrics({"claims": None}, has_ground_truth=False)

    def test_missing_claims_is_not_scored_as_perfectly_faithful(self):
        """'The judge told us nothing' is not evidence of faithfulness."""
        adapter = AnthropicEvaluator(model="claude-sonnet-5", api_key="test")

        with pytest.raises(EvaluationError):
            adapter._compute_metrics({"answer_relevance": "full"}, has_ground_truth=False)

    def test_valid_verdict_still_computes(self):
        adapter = AnthropicEvaluator(model="claude-sonnet-5", api_key="test")

        metrics = adapter._compute_metrics(
            {
                "claims": [
                    {"supported_by_context": True},
                    {"supported_by_context": False},
                ],
                "answer_relevance": "full",
                "contexts": [{"relevant_to_question": True}],
            },
            has_ground_truth=False,
        )

        assert metrics.faithfulness == pytest.approx(0.5)
        assert metrics.answer_relevancy == pytest.approx(1.0)

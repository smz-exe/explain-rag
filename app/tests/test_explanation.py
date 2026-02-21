"""Tests for explanation/faithfulness functionality."""

import pytest

from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness
from src.domain.entities.explanation import ClaimVerification
from tests.conftest import MockFaithfulnessPort


class TestFaithfulnessVerification:
    """Test faithfulness verification functionality."""

    @pytest.mark.asyncio
    async def test_verify_returns_result(self, sample_chunks):
        """Test that verify returns a FaithfulnessResult."""
        faithfulness = MockFaithfulnessPort()

        result = await faithfulness.verify(
            answer="Test answer",
            chunks=sample_chunks,
        )

        assert result is not None
        assert result.score is not None
        assert result.claims is not None

    @pytest.mark.asyncio
    async def test_faithfulness_all_supported(self, sample_chunks):
        """Test faithfulness with all supported claims."""
        claims = [
            ClaimVerification(
                claim="Claim 1",
                verdict="supported",
                evidence_chunk_ids=["chunk-001"],
                reasoning="Evidence found",
            ),
            ClaimVerification(
                claim="Claim 2",
                verdict="supported",
                evidence_chunk_ids=["chunk-002"],
                reasoning="Evidence found",
            ),
        ]
        faithfulness = MockFaithfulnessPort(score=1.0, claims=claims)

        result = await faithfulness.verify(answer="Answer", chunks=sample_chunks)

        assert result.score == 1.0
        assert all(c.verdict == "supported" for c in result.claims)

    @pytest.mark.asyncio
    async def test_faithfulness_mixed_verdicts(self, sample_chunks):
        """Test faithfulness with mixed verdicts."""
        claims = [
            ClaimVerification(
                claim="Supported",
                verdict="supported",
                evidence_chunk_ids=["chunk-001"],
                reasoning="Found",
            ),
            ClaimVerification(
                claim="Partial",
                verdict="partial",
                evidence_chunk_ids=["chunk-002"],
                reasoning="Partially found",
            ),
            ClaimVerification(
                claim="Unsupported",
                verdict="unsupported",
                evidence_chunk_ids=[],
                reasoning="Not found",
            ),
        ]
        # Score: (1.0 + 0.5 + 0.0) / 3 = 0.5
        faithfulness = MockFaithfulnessPort(score=0.5, claims=claims)

        result = await faithfulness.verify(answer="Answer", chunks=sample_chunks)

        assert result.score == 0.5
        verdicts = [c.verdict for c in result.claims]
        assert "supported" in verdicts
        assert "partial" in verdicts
        assert "unsupported" in verdicts

    @pytest.mark.asyncio
    async def test_verify_tracks_calls(self, sample_chunks):
        """Test that verify tracks its calls."""
        faithfulness = MockFaithfulnessPort()

        await faithfulness.verify(answer="Answer 1", chunks=sample_chunks)
        await faithfulness.verify(answer="Answer 2", chunks=sample_chunks)

        assert len(faithfulness.verify_calls) == 2


class TestFaithfulnessScoring:
    """Test faithfulness score calculation."""

    def test_calculate_score_all_supported(self):
        """Test score calculation with all supported claims."""
        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1",
                verdict="supported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
            ClaimVerification(
                claim="Claim 2",
                verdict="supported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
        ]

        score = adapter._calculate_score(results)
        assert score == 1.0

    def test_calculate_score_all_unsupported(self):
        """Test score calculation with all unsupported claims."""
        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1",
                verdict="unsupported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
            ClaimVerification(
                claim="Claim 2",
                verdict="unsupported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
        ]

        score = adapter._calculate_score(results)
        assert score == 0.0

    def test_calculate_score_mixed(self):
        """Test score calculation with mixed verdicts."""
        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1",
                verdict="supported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
            ClaimVerification(
                claim="Claim 2",
                verdict="partial",
                evidence_chunk_ids=[],
                reasoning="",
            ),
            ClaimVerification(
                claim="Claim 3",
                verdict="unsupported",
                evidence_chunk_ids=[],
                reasoning="",
            ),
        ]

        score = adapter._calculate_score(results)
        # (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert score == 0.5

    def test_calculate_score_empty(self):
        """Test score calculation with no claims."""
        adapter = LangChainFaithfulness()
        score = adapter._calculate_score([])
        assert score == 1.0

    def test_calculate_score_all_partial(self):
        """Test score calculation with all partial claims."""
        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1",
                verdict="partial",
                evidence_chunk_ids=[],
                reasoning="",
            ),
            ClaimVerification(
                claim="Claim 2",
                verdict="partial",
                evidence_chunk_ids=[],
                reasoning="",
            ),
        ]

        score = adapter._calculate_score(results)
        assert score == 0.5


class TestClaimVerification:
    """Test claim verification data structures."""

    def test_claim_verification_fields(self):
        """Test ClaimVerification has all required fields."""
        claim = ClaimVerification(
            claim="Test claim",
            verdict="supported",
            evidence_chunk_ids=["chunk-1", "chunk-2"],
            reasoning="Evidence found in chunks",
        )

        assert claim.claim == "Test claim"
        assert claim.verdict == "supported"
        assert claim.evidence_chunk_ids == ["chunk-1", "chunk-2"]
        assert claim.reasoning == "Evidence found in chunks"

    def test_claim_verification_verdict_values(self):
        """Test that verdicts are one of the expected values."""
        verdicts = ["supported", "partial", "unsupported"]

        for verdict in verdicts:
            claim = ClaimVerification(
                claim="Test",
                verdict=verdict,
                evidence_chunk_ids=[],
                reasoning="",
            )
            assert claim.verdict == verdict

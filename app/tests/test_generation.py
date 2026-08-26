"""Tests for LLM generation functionality."""

import pytest

from src.adapters.outbound.anthropic_rag import AnthropicRAG
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Citation
from tests.conftest import MockLLMPort


class TestCitationExtraction:
    """Test citation extraction from LLM responses."""

    def test_extract_citations_basic(self):
        """Test basic citation extraction."""
        adapter = AnthropicRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
            Chunk(id="chunk-2", paper_id="paper-1", content="Content 2", chunk_index=1),
        ]

        answer = "This is a fact [1]. Another fact [2]. Both together [1][2]."
        citations = adapter._extract_citations(answer, chunks)

        assert len(citations) == 3
        assert citations[0].chunk_ids == ["chunk-1"]
        assert citations[1].chunk_ids == ["chunk-2"]
        assert citations[2].chunk_ids == ["chunk-1", "chunk-2"]

    def test_extract_citations_no_citations(self):
        """Test extraction with no citations."""
        adapter = AnthropicRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
        ]

        answer = "This has no citations at all."
        citations = adapter._extract_citations(answer, chunks)

        assert len(citations) == 0

    def test_extract_citations_multiple_same(self):
        """Test extraction with multiple citations of same chunk."""
        adapter = AnthropicRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
        ]

        answer = "First point [1]. Second point also uses [1]."
        citations = adapter._extract_citations(answer, chunks)

        assert len(citations) == 2
        # Both should reference chunk-1
        for citation in citations:
            assert "chunk-1" in citation.chunk_ids

    def test_extract_citations_out_of_range(self):
        """Test extraction handles out-of-range citation numbers."""
        adapter = AnthropicRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
        ]

        # [5] is out of range (only 1 chunk)
        answer = "Valid citation [1]. Invalid citation [5]."
        citations = adapter._extract_citations(answer, chunks)

        # Should only have 1 valid citation
        assert len(citations) == 1
        assert citations[0].chunk_ids == ["chunk-1"]

    def test_extract_citations_sentence_splitting(self):
        """Test that citations are extracted with their sentences."""
        adapter = AnthropicRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
            Chunk(id="chunk-2", paper_id="paper-1", content="Content 2", chunk_index=1),
        ]

        answer = "Self-attention relates positions [1]. Transformers use it [2]."
        citations = adapter._extract_citations(answer, chunks)

        assert len(citations) == 2
        assert "Self-attention" in citations[0].claim
        assert "Transformers" in citations[1].claim


class TestInsufficientContext:
    """Test handling of insufficient context scenarios."""

    @pytest.mark.asyncio
    async def test_generate_with_custom_answer(self, sample_chunks):
        """Test that custom answer can be configured."""
        custom_answer = "Custom answer [1]."
        custom_citations = [Citation(claim="Custom claim", chunk_ids=["chunk-001"], confidence=0.8)]
        llm = MockLLMPort(answer=custom_answer, citations=custom_citations)

        result = await llm.generate(
            question="Any question",
            chunks=sample_chunks,
        )

        assert result.answer == custom_answer
        assert result.citations == custom_citations

"""Tests for query export functionality."""

import pytest

from src.adapters.inbound.http.query import _format_query_as_markdown
from src.domain.entities.explanation import (
    ClaimVerification,
    ExplanationTrace,
    FaithfulnessResult,
)
from src.domain.entities.query import Citation, QueryResponse, RetrievedChunk


@pytest.fixture
def sample_query_response() -> QueryResponse:
    """Create a sample query response for testing export."""
    return QueryResponse(
        query_id="test-query-123",
        question="What is self-attention?",
        answer="Self-attention is a mechanism [1]. It relates positions in a sequence [2].",
        citations=[
            Citation(
                claim="Self-attention is a mechanism",
                chunk_ids=["chunk-001"],
                confidence=0.9,
            ),
            Citation(
                claim="It relates positions in a sequence",
                chunk_ids=["chunk-002"],
                confidence=0.85,
            ),
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-001",
                paper_id="paper-001",
                paper_title="Attention Is All You Need",
                content="Self-attention, sometimes called intra-attention, is an attention mechanism.",
                similarity_score=0.92,
                rerank_score=0.95,
                original_rank=1,
                rank=1,
            ),
            RetrievedChunk(
                chunk_id="chunk-002",
                paper_id="paper-001",
                paper_title="Attention Is All You Need",
                content="It relates different positions of a single sequence to compute a representation.",
                similarity_score=0.88,
                rerank_score=None,
                original_rank=2,
                rank=2,
            ),
        ],
        faithfulness=FaithfulnessResult(
            score=0.9,
            claims=[
                ClaimVerification(
                    claim="Self-attention is a mechanism",
                    verdict="supported",
                    evidence_chunk_ids=["chunk-001"],
                    reasoning="Directly stated in chunk",
                ),
                ClaimVerification(
                    claim="It relates positions",
                    verdict="supported",
                    evidence_chunk_ids=["chunk-002"],
                    reasoning="Matches chunk content",
                ),
            ],
        ),
        trace=ExplanationTrace(
            embedding_time_ms=50.0,
            retrieval_time_ms=100.0,
            reranking_time_ms=150.0,
            generation_time_ms=2000.0,
            faithfulness_time_ms=1500.0,
            total_time_ms=3800.0,
        ),
    )


class TestFormatQueryAsMarkdown:
    """Test the markdown formatting function."""

    def test_includes_query_id(self, sample_query_response):
        """Test markdown includes query ID."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "test-query-123" in markdown

    def test_includes_question(self, sample_query_response):
        """Test markdown includes the question."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "What is self-attention?" in markdown

    def test_includes_answer(self, sample_query_response):
        """Test markdown includes the answer."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Self-attention is a mechanism [1]" in markdown

    def test_includes_chunks(self, sample_query_response):
        """Test markdown includes retrieved chunks."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Attention Is All You Need" in markdown
        assert "Self-attention, sometimes called intra-attention" in markdown

    def test_includes_scores(self, sample_query_response):
        """Test markdown includes similarity scores."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Similarity: 0.92" in markdown
        assert "Rerank: 0.95" in markdown

    def test_includes_faithfulness(self, sample_query_response):
        """Test markdown includes faithfulness score."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "90%" in markdown
        assert "SUPPORTED" in markdown

    def test_includes_timing(self, sample_query_response):
        """Test markdown includes performance timing."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Embedding: 50ms" in markdown
        assert "Total: 3800ms" in markdown

    def test_optional_reranking_time(self, sample_query_response):
        """Test reranking time is included when present."""
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Reranking: 150ms" in markdown

    def test_no_reranking_time_when_none(self, sample_query_response):
        """Test reranking time is omitted when None."""
        sample_query_response.trace.reranking_time_ms = None
        markdown = _format_query_as_markdown(sample_query_response)
        assert "Reranking:" not in markdown

    def test_chunk_content_truncated(self, sample_query_response):
        """Test long chunk content is truncated."""
        # Create a chunk with very long content
        sample_query_response.retrieved_chunks[0].content = "x" * 600
        markdown = _format_query_as_markdown(sample_query_response)
        # Content should be truncated to 500 chars + "..."
        assert "x" * 500 + "..." in markdown

    def test_chunk_without_rerank_score(self, sample_query_response):
        """Test chunk without rerank score doesn't show rerank."""
        markdown = _format_query_as_markdown(sample_query_response)
        # Second chunk has no rerank score
        assert "Similarity: 0.88" in markdown


@pytest.mark.asyncio
async def test_export_query_not_found(client):
    """Test export returns 404 for unknown query_id."""
    response = await client.get("/query/nonexistent-id/export")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_export_returns_markdown_content_type(client):
    """Test export returns correct content type header."""
    # First submit a query to create a stored response
    query_response = await client.post(
        "/query",
        json={"question": "What is the Transformer architecture?"},
    )
    assert query_response.status_code == 200
    query_id = query_response.json()["query_id"]

    # Now export it
    export_response = await client.get(f"/query/{query_id}/export")
    assert export_response.status_code == 200
    assert "text/markdown" in export_response.headers["content-type"]


@pytest.mark.asyncio
async def test_export_has_attachment_header(client):
    """Test export has Content-Disposition attachment header."""
    query_response = await client.post(
        "/query",
        json={"question": "What is attention?"},
    )
    assert query_response.status_code == 200
    query_id = query_response.json()["query_id"]

    export_response = await client.get(f"/query/{query_id}/export")
    assert export_response.status_code == 200
    assert "attachment" in export_response.headers["content-disposition"]
    assert f"query-{query_id[:8]}.md" in export_response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_export_contains_query_content(client):
    """Test exported markdown contains the query content."""
    question = "What is the purpose of multi-head attention?"
    query_response = await client.post(
        "/query",
        json={"question": question},
    )
    assert query_response.status_code == 200
    query_id = query_response.json()["query_id"]

    export_response = await client.get(f"/query/{query_id}/export")
    assert export_response.status_code == 200

    content = export_response.text
    assert "# Query Export" in content
    assert question in content
    assert "## Question" in content
    assert "## Answer" in content

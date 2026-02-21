import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.explanation import ClaimVerification


@pytest.mark.asyncio
async def test_query_endpoint_no_chunks(client):
    """Test query endpoint returns insufficient context when no papers ingested."""
    response = await client.post(
        "/query",
        json={"question": "What is hexagonal architecture?"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "query_id" in data
    assert data["question"] == "What is hexagonal architecture?"
    assert "cannot answer" in data["answer"].lower() or len(data["retrieved_chunks"]) == 0


@pytest.mark.asyncio
async def test_query_endpoint_validation(client):
    """Test query endpoint validates request."""
    # Missing question
    response = await client.post("/query", json={})
    assert response.status_code == 422

    # Invalid top_k
    response = await client.post(
        "/query",
        json={"question": "test", "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_explanation_not_found(client):
    """Test get explanation returns 404 for unknown query_id."""
    response = await client.get("/query/nonexistent-id/explanation")
    assert response.status_code == 404


class TestCitationExtraction:
    """Test citation extraction logic."""

    def test_extract_citations_basic(self):
        """Test basic citation extraction."""
        from src.adapters.outbound.langchain_rag import LangChainRAG

        adapter = LangChainRAG()
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
        from src.adapters.outbound.langchain_rag import LangChainRAG

        adapter = LangChainRAG()
        chunks = [
            Chunk(id="chunk-1", paper_id="paper-1", content="Content 1", chunk_index=0),
        ]

        answer = "This has no citations at all."
        citations = adapter._extract_citations(answer, chunks)

        assert len(citations) == 0


class TestFaithfulnessScoring:
    """Test faithfulness scoring logic."""

    def test_calculate_score_all_supported(self):
        """Test score calculation with all supported claims."""
        from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness

        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1", verdict="supported", evidence_chunk_ids=[], reasoning=""
            ),
            ClaimVerification(
                claim="Claim 2", verdict="supported", evidence_chunk_ids=[], reasoning=""
            ),
        ]

        score = adapter._calculate_score(results)
        assert score == 1.0

    def test_calculate_score_mixed(self):
        """Test score calculation with mixed verdicts."""
        from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness

        adapter = LangChainFaithfulness()
        results = [
            ClaimVerification(
                claim="Claim 1", verdict="supported", evidence_chunk_ids=[], reasoning=""
            ),
            ClaimVerification(
                claim="Claim 2", verdict="partial", evidence_chunk_ids=[], reasoning=""
            ),
            ClaimVerification(
                claim="Claim 3", verdict="unsupported", evidence_chunk_ids=[], reasoning=""
            ),
        ]

        score = adapter._calculate_score(results)
        assert score == 0.5  # (1.0 + 0.5 + 0.0) / 3

    def test_calculate_score_empty(self):
        """Test score calculation with no claims."""
        from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness

        adapter = LangChainFaithfulness()
        score = adapter._calculate_score([])
        assert score == 1.0

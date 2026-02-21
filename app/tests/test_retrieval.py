"""Tests for retrieval functionality."""

import pytest

from src.domain.entities.chunk import Chunk
from tests.conftest import MockEmbeddingPort, MockVectorStorePort


class TestVectorStoreSearch:
    """Test vector store search functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_top_k(self, sample_chunks):
        """Test that search returns the correct number of results."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)
        embedding = MockEmbeddingPort()

        query_embedding = await embedding.embed_query("What is attention?")
        results = await vector_store.search(query_embedding=query_embedding, top_k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_vector_search_with_paper_filter(self, sample_chunks):
        """Test that search filters by paper_id correctly."""
        # Add chunks from different papers
        chunks = sample_chunks + [
            Chunk(
                id="chunk-other",
                paper_id="paper-002",
                content="Content from another paper",
                chunk_index=0,
                metadata={"paper_title": "Other Paper"},
            )
        ]
        vector_store = MockVectorStorePort(chunks=chunks)
        embedding = MockEmbeddingPort()

        query_embedding = await embedding.embed_query("test query")
        results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=10,
            filter={"paper_id": {"$in": ["paper-001"]}},
        )

        # Should only return chunks from paper-001
        for chunk, _ in results:
            assert chunk.paper_id == "paper-001"

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self):
        """Test search with no chunks returns empty list."""
        vector_store = MockVectorStorePort(chunks=[])
        embedding = MockEmbeddingPort()

        query_embedding = await embedding.embed_query("test query")
        results = await vector_store.search(query_embedding=query_embedding, top_k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_similarity_scores_in_valid_range(self, sample_chunks):
        """Test that similarity scores are between 0 and 1."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)
        embedding = MockEmbeddingPort()

        query_embedding = await embedding.embed_query("test query")
        results = await vector_store.search(query_embedding=query_embedding, top_k=10)

        for _, score in results:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_results_ordered_by_score(self, sample_chunks):
        """Test that results are ordered by score descending."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)
        embedding = MockEmbeddingPort()

        query_embedding = await embedding.embed_query("test query")
        results = await vector_store.search(query_embedding=query_embedding, top_k=10)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestEmbedding:
    """Test embedding functionality."""

    @pytest.mark.asyncio
    async def test_embed_texts_returns_correct_count(self):
        """Test that embed_texts returns embeddings for all texts."""
        embedding = MockEmbeddingPort()
        texts = ["text 1", "text 2", "text 3"]

        embeddings = await embedding.embed_texts(texts)

        assert len(embeddings) == 3

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_embedding(self):
        """Test that embed_query returns a single embedding."""
        embedding = MockEmbeddingPort()

        result = await embedding.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == embedding.embedding_dim

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        embedding = MockEmbeddingPort(embedding_dim=384)
        texts = ["text 1", "text 2"]

        embeddings = await embedding.embed_texts(texts)

        for emb in embeddings:
            assert len(emb) == 384


class TestVectorStoreManagement:
    """Test vector store management operations."""

    @pytest.mark.asyncio
    async def test_add_chunks(self, sample_chunks):
        """Test adding chunks to vector store."""
        vector_store = MockVectorStorePort()

        await vector_store.add_chunks(sample_chunks)

        assert len(vector_store.chunks) == 3
        assert len(vector_store.added_chunks) == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, sample_chunks):
        """Test getting vector store stats."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)

        stats = await vector_store.get_stats()

        assert "total_chunks" in stats
        assert stats["total_chunks"] == 3

    @pytest.mark.asyncio
    async def test_list_papers(self, sample_chunks):
        """Test listing papers in vector store."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)

        papers = await vector_store.list_papers()

        assert len(papers) >= 1

    @pytest.mark.asyncio
    async def test_delete_paper(self, sample_chunks):
        """Test deleting a paper's chunks."""
        vector_store = MockVectorStorePort(chunks=sample_chunks)

        deleted_count = await vector_store.delete_paper("paper-001")

        assert deleted_count == 3
        assert len(vector_store.chunks) == 0

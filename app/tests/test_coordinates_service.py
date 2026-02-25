"""Tests for CoordinatesService."""

import pytest

from src.application.coordinates_service import CoordinatesService
from tests.conftest import (
    MockClusteringPort,
    MockDimensionalityReductionPort,
    MockVectorStorePort,
)


class TestCoordinatesService:
    """Test CoordinatesService functionality."""

    @pytest.fixture
    def mock_vector_store_with_papers(self, sample_chunks):
        """Create a mock vector store with paper data."""
        store = MockVectorStorePort(chunks=sample_chunks)
        # Override list_papers to return proper metadata
        async def mock_list_papers():
            return [
                {
                    "paper_id": "paper-001",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "chunk_count": 3,
                },
            ]
        store.list_papers = mock_list_papers
        return store

    @pytest.fixture
    def service(self, mock_vector_store_with_papers):
        """Create a CoordinatesService with mock adapters."""
        return CoordinatesService(
            vector_store=mock_vector_store_with_papers,
            dim_reducer=MockDimensionalityReductionPort(),
            clusterer=MockClusteringPort(),
        )

    @pytest.mark.asyncio
    async def test_is_computed_false_initially(self, service):
        """Test that is_computed is False before recompute."""
        assert service.is_computed is False
        assert service.computed_at is None

    @pytest.mark.asyncio
    async def test_get_paper_coordinates_empty_initially(self, service):
        """Test that coordinates are empty before recompute."""
        coords = await service.get_paper_coordinates()
        assert coords == []

    @pytest.mark.asyncio
    async def test_get_clusters_empty_initially(self, service):
        """Test that clusters are empty before recompute."""
        clusters = await service.get_clusters()
        assert clusters == []

    @pytest.mark.asyncio
    async def test_recompute_all_processes_papers(self, service):
        """Test that recompute_all processes papers correctly."""
        result = await service.recompute_all()

        assert result["papers_processed"] == 1
        assert "clusters_found" in result
        assert "time_ms" in result
        assert service.is_computed is True
        assert service.computed_at is not None

    @pytest.mark.asyncio
    async def test_recompute_all_creates_coordinates(self, service):
        """Test that recompute creates paper coordinates."""
        await service.recompute_all()

        coords = await service.get_paper_coordinates()
        assert len(coords) == 1

        coord = coords[0]
        assert coord.paper_id == "paper-001"
        assert coord.arxiv_id == "1706.03762"
        assert coord.title == "Attention Is All You Need"
        assert len(coord.coords) == 3
        assert coord.chunk_count == 3

    @pytest.mark.asyncio
    async def test_recompute_all_creates_clusters(self, service):
        """Test that recompute creates clusters."""
        await service.recompute_all()

        clusters = await service.get_clusters()
        # With one paper and mock clustering, we get clusters based on mock behavior
        assert isinstance(clusters, list)

    @pytest.mark.asyncio
    async def test_get_query_coordinates_requires_fitting(self, service):
        """Test that query coordinates require prior fitting."""
        # Before recompute, reducer is not fitted
        coords = await service.get_query_coordinates([0.5] * 384)
        assert coords is None

    @pytest.mark.asyncio
    async def test_get_query_coordinates_after_recompute(self, service):
        """Test query coordinates after recompute."""
        await service.recompute_all()

        coords = await service.get_query_coordinates([0.5] * 384)
        assert coords is not None
        assert len(coords) == 3

    @pytest.mark.asyncio
    async def test_clear_cache(self, service):
        """Test that clear_cache resets the service state."""
        await service.recompute_all()
        assert service.is_computed is True

        service.clear_cache()

        assert service.is_computed is False
        assert service.computed_at is None
        coords = await service.get_paper_coordinates()
        assert coords == []

    @pytest.mark.asyncio
    async def test_recompute_with_empty_store(self):
        """Test recompute with empty vector store."""
        empty_store = MockVectorStorePort(chunks=[])
        service = CoordinatesService(
            vector_store=empty_store,
            dim_reducer=MockDimensionalityReductionPort(),
            clusterer=MockClusteringPort(),
        )

        result = await service.recompute_all()

        assert result["papers_processed"] == 0
        assert result["clusters_found"] == 0
        # is_computed is False when no papers (nothing to compute)
        assert service.is_computed is False
        # But computed_at is set to indicate computation was attempted
        assert service.computed_at is not None

    @pytest.mark.asyncio
    async def test_coordinates_are_copied_not_referenced(self, service):
        """Test that get_paper_coordinates returns a copy."""
        await service.recompute_all()

        coords1 = await service.get_paper_coordinates()
        coords2 = await service.get_paper_coordinates()

        # Should be equal but different objects
        assert coords1 == coords2
        assert coords1 is not coords2


class TestClusterLabelGeneration:
    """Test cluster label generation logic."""

    @pytest.fixture
    def service(self, sample_chunks):
        """Create a CoordinatesService for testing."""
        store = MockVectorStorePort(chunks=sample_chunks)
        return CoordinatesService(
            vector_store=store,
            dim_reducer=MockDimensionalityReductionPort(),
            clusterer=MockClusteringPort(),
        )

    def test_generate_label_from_titles(self, service):
        """Test label generation from paper titles."""
        titles = [
            "Deep Learning for Natural Language Processing",
            "Natural Language Understanding with Transformers",
            "Advances in Natural Language Generation",
        ]

        label = service._generate_cluster_label(titles)

        # Should contain common significant word
        assert "natural" in label.lower() or "language" in label.lower()

    def test_generate_label_empty_titles(self, service):
        """Test label generation with empty titles."""
        label = service._generate_cluster_label([])
        assert label == "Uncategorized"

    def test_generate_label_single_title(self, service):
        """Test label generation with single title."""
        titles = ["Attention Is All You Need"]

        label = service._generate_cluster_label(titles)

        # Should return something meaningful
        assert len(label) > 0
        assert label != "Uncategorized"

    def test_generate_label_filters_stop_words(self, service):
        """Test that stop words are filtered from labels."""
        titles = [
            "The Impact of the Model",
            "A Study of the Method",
        ]

        label = service._generate_cluster_label(titles)

        # Should not contain stop words
        label_lower = label.lower()
        assert "the" not in label_lower.split()
        assert "of" not in label_lower.split()


class TestMultiplePapers:
    """Test with multiple papers."""

    @pytest.fixture
    def multi_paper_service(self, sample_chunks):
        """Create service with multiple papers."""
        from src.domain.entities.chunk import Chunk

        # Create chunks for multiple papers
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                paper_id=f"paper-{i // 2}",
                content=f"Content for paper {i // 2}",
                chunk_index=i % 2,
            )
            for i in range(6)
        ]

        store = MockVectorStorePort(chunks=chunks)

        # Override list_papers
        async def mock_list_papers():
            return [
                {"paper_id": "paper-0", "arxiv_id": "2401.00001", "title": "Machine Learning Basics", "chunk_count": 2},
                {"paper_id": "paper-1", "arxiv_id": "2401.00002", "title": "Deep Learning Advances", "chunk_count": 2},
                {"paper_id": "paper-2", "arxiv_id": "2401.00003", "title": "Neural Network Training", "chunk_count": 2},
            ]
        store.list_papers = mock_list_papers

        # Override get_paper_embeddings
        async def mock_get_paper_embeddings():
            return [
                ("paper-0", [0.1] * 384),
                ("paper-1", [0.2] * 384),
                ("paper-2", [0.3] * 384),
            ]
        store.get_paper_embeddings = mock_get_paper_embeddings

        return CoordinatesService(
            vector_store=store,
            dim_reducer=MockDimensionalityReductionPort(),
            clusterer=MockClusteringPort(cluster_labels=[0, 0, 1]),  # 2 clusters
        )

    @pytest.mark.asyncio
    async def test_multiple_papers_processed(self, multi_paper_service):
        """Test processing multiple papers."""
        result = await multi_paper_service.recompute_all()

        assert result["papers_processed"] == 3

        coords = await multi_paper_service.get_paper_coordinates()
        assert len(coords) == 3

    @pytest.mark.asyncio
    async def test_clusters_group_papers(self, multi_paper_service):
        """Test that clusters group papers correctly."""
        await multi_paper_service.recompute_all()

        clusters = await multi_paper_service.get_clusters()

        # Should have 2 clusters based on mock labels [0, 0, 1]
        assert len(clusters) == 2

        # Find cluster 0 (should have paper-0 and paper-1)
        cluster_0 = next((c for c in clusters if c.id == 0), None)
        assert cluster_0 is not None
        assert len(cluster_0.paper_ids) == 2
        assert "paper-0" in cluster_0.paper_ids
        assert "paper-1" in cluster_0.paper_ids

        # Find cluster 1 (should have paper-2)
        cluster_1 = next((c for c in clusters if c.id == 1), None)
        assert cluster_1 is not None
        assert len(cluster_1.paper_ids) == 1
        assert "paper-2" in cluster_1.paper_ids

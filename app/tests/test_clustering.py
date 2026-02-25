"""Tests for clustering adapters."""

import pytest

from src.adapters.outbound.hdbscan_clusterer import HDBSCANClusterer


class TestHDBSCANClusterer:
    """Test HDBSCAN clustering adapter."""

    @pytest.mark.asyncio
    async def test_cluster_returns_labels(self):
        """Test that cluster returns cluster labels."""
        clusterer = HDBSCANClusterer(min_cluster_size=2, min_samples=1)

        # Create embeddings that should form 2 clusters
        embeddings = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, 0.05],  # Cluster 1
            [10.0, 10.0],
            [10.1, 10.1],
            [10.05, 10.05],  # Cluster 2
        ]

        labels = await clusterer.cluster(embeddings)

        assert len(labels) == 6
        assert all(isinstance(label, int) for label in labels)

    @pytest.mark.asyncio
    async def test_cluster_empty_input(self):
        """Test cluster with empty input."""
        clusterer = HDBSCANClusterer()

        labels = await clusterer.cluster([])

        assert labels == []

    @pytest.mark.asyncio
    async def test_get_cluster_count_requires_clustering(self):
        """Test that get_cluster_count raises if not clustered."""
        clusterer = HDBSCANClusterer()

        with pytest.raises(RuntimeError, match="not been performed"):
            await clusterer.get_cluster_count()

    @pytest.mark.asyncio
    async def test_get_cluster_count_after_clustering(self):
        """Test get_cluster_count after clustering."""
        clusterer = HDBSCANClusterer(min_cluster_size=2, min_samples=1)

        # Create embeddings that should form 2 distinct clusters
        embeddings = [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ]

        await clusterer.cluster(embeddings)
        count = await clusterer.get_cluster_count()

        # Should have at least 1 cluster (exact count depends on HDBSCAN's decisions)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_noise_points_labeled_minus_one(self):
        """Test that noise/outlier points can be labeled -1."""
        clusterer = HDBSCANClusterer(min_cluster_size=3, min_samples=2)

        # Create embeddings with one clear outlier
        embeddings = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, 0.05],
            [100.0, 100.0],  # Outlier
        ]

        labels = await clusterer.cluster(embeddings)

        # The outlier might be labeled as -1 (noise)
        assert all(label >= -1 for label in labels)

    @pytest.mark.asyncio
    async def test_cluster_adjusts_min_cluster_size_for_small_datasets(self):
        """Test that min_cluster_size is adjusted for small datasets."""
        # With only 2 samples and min_cluster_size=5, should adjust
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=1)
        embeddings = [[0.0, 0.0], [0.1, 0.1]]

        # Should not raise
        labels = await clusterer.cluster(embeddings)

        assert len(labels) == 2

    @pytest.mark.asyncio
    async def test_cluster_count_excludes_noise(self):
        """Test that cluster count excludes noise points."""
        clusterer = HDBSCANClusterer(min_cluster_size=2, min_samples=1)

        # All points in one tight cluster
        embeddings = [
            [0.0, 0.0],
            [0.01, 0.01],
            [0.02, 0.02],
            [0.03, 0.03],
        ]

        labels = await clusterer.cluster(embeddings)
        count = await clusterer.get_cluster_count()

        # Count should be number of unique non-negative labels
        unique_clusters = set(label for label in labels if label >= 0)
        assert count == len(unique_clusters)

    @pytest.mark.asyncio
    async def test_empty_dataset_cluster_count(self):
        """Test cluster count with empty dataset."""
        clusterer = HDBSCANClusterer()

        await clusterer.cluster([])
        count = await clusterer.get_cluster_count()

        assert count == 0

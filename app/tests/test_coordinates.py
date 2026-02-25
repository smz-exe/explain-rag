"""Tests for coordinates domain entities."""

import pytest

from src.domain.entities.coordinates import Cluster, PaperCoordinates


class TestPaperCoordinates:
    """Test PaperCoordinates entity."""

    def test_create_with_required_fields(self):
        """Test creating PaperCoordinates with required fields."""
        coords = PaperCoordinates(
            paper_id="paper-001",
            arxiv_id="2401.12345",
            title="Test Paper",
            coords=(0.5, -0.3, 0.8),
        )

        assert coords.paper_id == "paper-001"
        assert coords.arxiv_id == "2401.12345"
        assert coords.title == "Test Paper"
        assert coords.coords == (0.5, -0.3, 0.8)
        assert coords.cluster_id is None
        assert coords.chunk_count == 0

    def test_create_with_all_fields(self):
        """Test creating PaperCoordinates with all fields."""
        coords = PaperCoordinates(
            paper_id="paper-002",
            arxiv_id="2401.67890",
            title="Another Paper",
            coords=(1.0, 2.0, 3.0),
            cluster_id=2,
            chunk_count=150,
        )

        assert coords.cluster_id == 2
        assert coords.chunk_count == 150

    def test_noise_cluster_id(self):
        """Test that cluster_id can be -1 for noise points."""
        coords = PaperCoordinates(
            paper_id="paper-003",
            arxiv_id="2401.11111",
            title="Noise Paper",
            coords=(0.0, 0.0, 0.0),
            cluster_id=-1,
        )

        assert coords.cluster_id == -1

    def test_coords_tuple_access(self):
        """Test accessing individual coordinates."""
        coords = PaperCoordinates(
            paper_id="paper-004",
            arxiv_id="2401.22222",
            title="Coords Test",
            coords=(1.5, -2.5, 3.5),
        )

        x, y, z = coords.coords
        assert x == 1.5
        assert y == -2.5
        assert z == 3.5

    def test_serialization(self):
        """Test that entity can be serialized to dict."""
        coords = PaperCoordinates(
            paper_id="paper-005",
            arxiv_id="2401.33333",
            title="Serialization Test",
            coords=(0.1, 0.2, 0.3),
            cluster_id=1,
            chunk_count=50,
        )

        data = coords.model_dump()

        assert data["paper_id"] == "paper-005"
        assert data["coords"] == (0.1, 0.2, 0.3)
        assert data["cluster_id"] == 1


class TestCluster:
    """Test Cluster entity."""

    def test_create_cluster(self):
        """Test creating a Cluster."""
        cluster = Cluster(
            id=0,
            label="Machine Learning",
            paper_ids=["paper-001", "paper-002", "paper-003"],
        )

        assert cluster.id == 0
        assert cluster.label == "Machine Learning"
        assert len(cluster.paper_ids) == 3
        assert "paper-001" in cluster.paper_ids

    def test_empty_cluster(self):
        """Test creating a cluster with no papers."""
        cluster = Cluster(
            id=1,
            label="Empty Cluster",
            paper_ids=[],
        )

        assert cluster.paper_ids == []

    def test_serialization(self):
        """Test that cluster can be serialized to dict."""
        cluster = Cluster(
            id=2,
            label="Natural Language Processing",
            paper_ids=["paper-a", "paper-b"],
        )

        data = cluster.model_dump()

        assert data["id"] == 2
        assert data["label"] == "Natural Language Processing"
        assert data["paper_ids"] == ["paper-a", "paper-b"]

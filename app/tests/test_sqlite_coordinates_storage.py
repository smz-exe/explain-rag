"""Tests for coordinates storage functionality."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.adapters.outbound.sqlite_coordinates_storage import SQLiteCoordinatesStorage
from src.domain.entities.coordinates import Cluster, PaperCoordinates
from tests.conftest import MockCoordinatesStoragePort


@pytest.fixture
def sample_coordinates() -> list[PaperCoordinates]:
    """Create sample coordinates for testing."""
    return [
        PaperCoordinates(
            paper_id="paper-001",
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            coords=(1.0, 2.0, 3.0),
            cluster_id=0,
            chunk_count=10,
        ),
        PaperCoordinates(
            paper_id="paper-002",
            arxiv_id="1810.04805",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            coords=(4.0, 5.0, 6.0),
            cluster_id=0,
            chunk_count=15,
        ),
        PaperCoordinates(
            paper_id="paper-003",
            arxiv_id="2005.14165",
            title="Language Models are Few-Shot Learners",
            coords=(7.0, 8.0, 9.0),
            cluster_id=None,  # Noise point
            chunk_count=20,
        ),
    ]


@pytest.fixture
def sample_clusters() -> list[Cluster]:
    """Create sample clusters for testing."""
    return [
        Cluster(
            id=0,
            label="Attention & Transformers",
            paper_ids=["paper-001", "paper-002"],
        ),
    ]


@pytest.fixture
def sample_computed_at() -> datetime:
    """Create a sample computed_at timestamp."""
    return datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


class TestMockCoordinatesStorage:
    """Test the mock coordinates storage adapter."""

    @pytest.mark.asyncio
    async def test_load_empty(self):
        """Test loading from empty storage."""
        storage = MockCoordinatesStoragePort()

        coords, clusters, computed_at = await storage.load()

        assert coords == []
        assert clusters == []
        assert computed_at is None
        assert storage.load_calls == 1

    @pytest.mark.asyncio
    async def test_save_and_load(self, sample_coordinates, sample_clusters, sample_computed_at):
        """Test saving and loading coordinates."""
        storage = MockCoordinatesStoragePort()

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        coords, clusters, computed_at = await storage.load()

        assert len(coords) == 3
        assert len(clusters) == 1
        assert computed_at == sample_computed_at
        assert len(storage.save_calls) == 1

    @pytest.mark.asyncio
    async def test_clear(self, sample_coordinates, sample_clusters, sample_computed_at):
        """Test clearing storage."""
        storage = MockCoordinatesStoragePort(
            initial_coordinates=sample_coordinates,
            initial_clusters=sample_clusters,
            initial_computed_at=sample_computed_at,
        )

        await storage.clear()
        coords, clusters, computed_at = await storage.load()

        assert coords == []
        assert clusters == []
        assert computed_at is None
        assert storage.clear_calls == 1


class TestSQLiteCoordinatesStorage:
    """Test the SQLite coordinates storage adapter."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_coordinates.db"

    @pytest.mark.asyncio
    async def test_load_empty(self, temp_db_path):
        """Test loading from empty database."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        coords, clusters, computed_at = await storage.load()

        assert coords == []
        assert clusters == []
        assert computed_at is None

    @pytest.mark.asyncio
    async def test_save_and_load_coordinates(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test saving and loading coordinates."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        coords, clusters, computed_at = await storage.load()

        assert len(coords) == 3
        assert coords[0].paper_id == "paper-001"
        assert coords[0].arxiv_id == "1706.03762"
        assert coords[0].title == "Attention Is All You Need"
        assert coords[0].coords == (1.0, 2.0, 3.0)
        assert coords[0].cluster_id == 0
        assert coords[0].chunk_count == 10

    @pytest.mark.asyncio
    async def test_save_and_load_clusters(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test saving and loading clusters."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        coords, clusters, computed_at = await storage.load()

        assert len(clusters) == 1
        assert clusters[0].id == 0
        assert clusters[0].label == "Attention & Transformers"
        assert clusters[0].paper_ids == ["paper-001", "paper-002"]

    @pytest.mark.asyncio
    async def test_computed_at_preserved(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test that computed_at timestamp is preserved."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        _, _, computed_at = await storage.load()

        assert computed_at == sample_computed_at

    @pytest.mark.asyncio
    async def test_handles_null_cluster_id(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test that null cluster_id (noise points) is handled correctly."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        coords, _, _ = await storage.load()

        # paper-003 has cluster_id=None
        paper_003 = next(c for c in coords if c.paper_id == "paper-003")
        assert paper_003.cluster_id is None

    @pytest.mark.asyncio
    async def test_save_replaces_existing(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test that saving replaces existing data."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        # Save initial data
        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)

        # Save new data
        new_coords = [
            PaperCoordinates(
                paper_id="paper-new",
                arxiv_id="2024.12345",
                title="New Paper",
                coords=(10.0, 20.0, 30.0),
                cluster_id=1,
                chunk_count=5,
            ),
        ]
        new_clusters = [
            Cluster(id=1, label="New Cluster", paper_ids=["paper-new"]),
        ]
        new_computed_at = datetime(2025, 2, 1, 12, 0, 0, tzinfo=UTC)

        await storage.save(new_coords, new_clusters, new_computed_at)
        coords, clusters, computed_at = await storage.load()

        # Should only have new data
        assert len(coords) == 1
        assert coords[0].paper_id == "paper-new"
        assert len(clusters) == 1
        assert clusters[0].label == "New Cluster"
        assert computed_at == new_computed_at

    @pytest.mark.asyncio
    async def test_clear(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test clearing all data."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)

        await storage.save(sample_coordinates, sample_clusters, sample_computed_at)
        await storage.clear()
        coords, clusters, computed_at = await storage.load()

        assert coords == []
        assert clusters == []
        assert computed_at is None

    @pytest.mark.asyncio
    async def test_persistence_across_instances(
        self, temp_db_path, sample_coordinates, sample_clusters, sample_computed_at
    ):
        """Test that data persists across storage instances."""
        # Save with first instance
        storage1 = SQLiteCoordinatesStorage(db_path=temp_db_path)
        await storage1.save(sample_coordinates, sample_clusters, sample_computed_at)

        # Load with second instance
        storage2 = SQLiteCoordinatesStorage(db_path=temp_db_path)
        coords, clusters, computed_at = await storage2.load()

        assert len(coords) == 3
        assert len(clusters) == 1
        assert computed_at == sample_computed_at

    @pytest.mark.asyncio
    async def test_empty_save(self, temp_db_path):
        """Test saving empty data."""
        storage = SQLiteCoordinatesStorage(db_path=temp_db_path)
        computed_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

        await storage.save([], [], computed_at)
        coords, clusters, loaded_computed_at = await storage.load()

        assert coords == []
        assert clusters == []
        # computed_at is extracted from rows, so it will be None if no rows
        assert loaded_computed_at is None

"""SQLite-based coordinates storage adapter."""

import json
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

from src.domain.entities.coordinates import Cluster, PaperCoordinates
from src.domain.ports.coordinates_storage import CoordinatesStoragePort

logger = logging.getLogger(__name__)


class SQLiteCoordinatesStorage(CoordinatesStoragePort):
    """SQLite-based coordinates storage adapter."""

    def __init__(self, db_path: str | Path = "./data/queries.db"):
        """Initialize the SQLite coordinates storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database and tables exist."""
        if self._initialized:
            return

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS paper_coordinates (
                    paper_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    cluster_id INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    computed_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    paper_ids TEXT NOT NULL,
                    computed_at TEXT NOT NULL
                )
            """)
            await db.commit()

        self._initialized = True
        logger.info(f"SQLite coordinates storage initialized at {self._db_path}")

    async def load(
        self,
    ) -> tuple[list[PaperCoordinates], list[Cluster], datetime | None]:
        """Load stored coordinates and clusters."""
        await self._ensure_initialized()

        coordinates: list[PaperCoordinates] = []
        clusters: list[Cluster] = []
        computed_at: datetime | None = None

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # Load coordinates
            async with db.execute(
                "SELECT * FROM paper_coordinates ORDER BY paper_id"
            ) as cursor:
                rows = await cursor.fetchall()

            for row in rows:
                coordinates.append(
                    PaperCoordinates(
                        paper_id=row["paper_id"],
                        arxiv_id=row["arxiv_id"],
                        title=row["title"],
                        coords=(row["x"], row["y"], row["z"]),
                        cluster_id=row["cluster_id"],
                        chunk_count=row["chunk_count"],
                    )
                )
                # Get computed_at from first row
                if computed_at is None and row["computed_at"]:
                    computed_at = datetime.fromisoformat(row["computed_at"])

            # Load clusters
            async with db.execute("SELECT * FROM clusters ORDER BY id") as cursor:
                rows = await cursor.fetchall()

            for row in rows:
                paper_ids = json.loads(row["paper_ids"])
                clusters.append(
                    Cluster(
                        id=row["id"],
                        label=row["label"],
                        paper_ids=paper_ids,
                    )
                )
                # Get computed_at if not already set
                if computed_at is None and row["computed_at"]:
                    computed_at = datetime.fromisoformat(row["computed_at"])

        if coordinates:
            logger.info(
                f"Loaded {len(coordinates)} coordinates and {len(clusters)} clusters"
            )

        return coordinates, clusters, computed_at

    async def save(
        self,
        coordinates: list[PaperCoordinates],
        clusters: list[Cluster],
        computed_at: datetime,
    ) -> None:
        """Save coordinates and clusters, replacing any existing data."""
        await self._ensure_initialized()

        computed_at_iso = computed_at.isoformat()

        try:
            async with aiosqlite.connect(self._db_path) as db:
                # Clear existing data
                await db.execute("DELETE FROM paper_coordinates")
                await db.execute("DELETE FROM clusters")

                # Insert coordinates
                if coordinates:
                    await db.executemany(
                        """
                        INSERT INTO paper_coordinates
                        (paper_id, arxiv_id, title, x, y, z, cluster_id, chunk_count, computed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                c.paper_id,
                                c.arxiv_id,
                                c.title,
                                c.coords[0],
                                c.coords[1],
                                c.coords[2],
                                c.cluster_id,
                                c.chunk_count,
                                computed_at_iso,
                            )
                            for c in coordinates
                        ],
                    )

                # Insert clusters
                if clusters:
                    await db.executemany(
                        """
                        INSERT INTO clusters (id, label, paper_ids, computed_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                cluster.id,
                                cluster.label,
                                json.dumps(cluster.paper_ids),
                                computed_at_iso,
                            )
                            for cluster in clusters
                        ],
                    )

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to save coordinates to database: {e}")
            raise

        logger.info(
            f"Saved {len(coordinates)} coordinates and {len(clusters)} clusters"
        )

    async def clear(self) -> None:
        """Clear all stored coordinates and clusters."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM paper_coordinates")
            await db.execute("DELETE FROM clusters")
            await db.commit()

        logger.info("Cleared all coordinates and clusters from storage")

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from src.domain.entities.query import QueryResponse
from src.domain.ports.query_storage import QueryStoragePort

logger = logging.getLogger(__name__)


class SQLiteQueryStorage(QueryStoragePort):
    """SQLite-based query storage adapter."""

    def __init__(self, db_path: str | Path = "./data/queries.db"):
        """Initialize the SQLite query storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database and table exist."""
        if self._initialized:
            return

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer_preview TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_created_at
                ON queries(created_at DESC)
            """)
            await db.commit()

        self._initialized = True
        logger.info(f"SQLite query storage initialized at {self._db_path}")

    async def store(self, response: QueryResponse) -> None:
        """Store a query response."""
        await self._ensure_initialized()

        created_at = datetime.now(UTC).isoformat()
        response_json = response.model_dump_json()
        answer_preview = response.answer[:200] if response.answer else ""

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO queries
                (id, response_json, question, answer_preview, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    response.query_id,
                    response_json,
                    response.question,
                    answer_preview,
                    created_at,
                ),
            )
            await db.commit()

        logger.debug(f"Stored query {response.query_id}")

    async def get(self, query_id: str) -> QueryResponse | None:
        """Retrieve a query response by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT response_json FROM queries WHERE id = ?",
                (query_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        data = json.loads(row["response_json"])
        return QueryResponse.model_validate(data)

    async def list_recent(self, limit: int = 20) -> list[dict]:
        """List recent queries with summary information."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, question, answer_preview, created_at
                FROM queries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "query_id": row["id"],
                "question": row["question"],
                "answer_preview": row["answer_preview"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def delete(self, query_id: str) -> bool:
        """Delete a query from storage."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM queries WHERE id = ?",
                (query_id,),
            )
            await db.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug(f"Deleted query {query_id}")

        return deleted

    async def count(self) -> int:
        """Get the total number of stored queries."""
        await self._ensure_initialized()

        async with (
            aiosqlite.connect(self._db_path) as db,
            db.execute("SELECT COUNT(*) FROM queries") as cursor,
        ):
            row = await cursor.fetchone()
            return row[0] if row else 0

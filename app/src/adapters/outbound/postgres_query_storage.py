"""PostgreSQL query storage adapter for storing query history and results."""

import json
import logging
from datetime import UTC, datetime

import asyncpg
from pgvector.asyncpg import register_vector

from src.domain.entities.query import QueryResponse
from src.domain.ports.query_storage import QueryStoragePort

logger = logging.getLogger(__name__)


class PostgresQueryStorage(QueryStoragePort):
    """PostgreSQL-based query storage adapter."""

    def __init__(
        self,
        database_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """Initialize the PostgreSQL query storage.

        Args:
            database_url: PostgreSQL connection URL.
            pool_min_size: Minimum connection pool size.
            pool_max_size: Maximum connection pool size.
        """
        self._database_url = database_url
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                init=self._init_connection,
                # Disable statement cache for pgbouncer compatibility (Supabase)
                statement_cache_size=0,
            )
            logger.info("PostgreSQL query storage pool created")
        return self._pool

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize a connection with pgvector support."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL query storage pool closed")

    async def store(self, response: QueryResponse) -> None:
        """Store a query response."""
        pool = await self._get_pool()

        created_at = datetime.now(UTC)

        # Serialize complex fields to JSON
        citations_json = json.dumps([c.model_dump() for c in response.citations])
        retrieved_chunks_json = json.dumps([c.model_dump() for c in response.retrieved_chunks])
        faithfulness_details_json = json.dumps(response.faithfulness.model_dump())
        timing_json = json.dumps(response.trace.model_dump())

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO queries (
                    id, question, answer, citations, retrieved_chunks,
                    faithfulness_score, faithfulness_details, timing, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    answer = EXCLUDED.answer,
                    citations = EXCLUDED.citations,
                    retrieved_chunks = EXCLUDED.retrieved_chunks,
                    faithfulness_score = EXCLUDED.faithfulness_score,
                    faithfulness_details = EXCLUDED.faithfulness_details,
                    timing = EXCLUDED.timing
                """,
                response.query_id,
                response.question,
                response.answer,
                citations_json,
                retrieved_chunks_json,
                response.faithfulness.score,
                faithfulness_details_json,
                timing_json,
                created_at,
            )

        logger.debug(f"Stored query {response.query_id}")

    async def get(self, query_id: str) -> QueryResponse | None:
        """Retrieve a query response by ID."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, question, answer, citations, retrieved_chunks,
                    faithfulness_score, faithfulness_details, timing
                FROM queries
                WHERE id = $1
                """,
                query_id,
            )

        if row is None:
            return None

        # Deserialize JSON fields
        citations = json.loads(row["citations"]) if row["citations"] else []
        retrieved_chunks = json.loads(row["retrieved_chunks"]) if row["retrieved_chunks"] else []
        faithfulness_details = (
            json.loads(row["faithfulness_details"]) if row["faithfulness_details"] else {}
        )
        timing = json.loads(row["timing"]) if row["timing"] else {}

        # Reconstruct QueryResponse
        return QueryResponse(
            query_id=str(row["id"]),
            question=row["question"],
            answer=row["answer"] or "",
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            faithfulness=faithfulness_details,
            trace=timing,
        )

    async def list_recent(self, limit: int = 20) -> list[dict]:
        """List recent queries with summary information."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, question, answer, created_at
                FROM queries
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        return [
            {
                "query_id": str(row["id"]),
                "question": row["question"],
                "answer_preview": (row["answer"][:200] if row["answer"] else ""),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    async def delete(self, query_id: str) -> bool:
        """Delete a query from storage."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM queries WHERE id = $1",
                query_id,
            )
            deleted = result == "DELETE 1"

        if deleted:
            logger.debug(f"Deleted query {query_id}")

        return deleted

    async def count(self) -> int:
        """Get the total number of stored queries."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM queries")

        return count or 0

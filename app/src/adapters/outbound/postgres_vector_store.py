"""PostgreSQL vector store adapter using pgvector for similarity search."""

import asyncio
import json
import logging
from collections import defaultdict

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from src.domain.entities.chunk import Chunk
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


def _sanitize_text(text: str | None) -> str:
    """Remove null bytes and other problematic characters from text.

    PostgreSQL text columns don't allow null bytes (0x00).
    This can happen when PDF extraction produces binary artifacts.
    """
    if text is None:
        return ""
    # Remove null bytes that cause PostgreSQL errors
    return text.replace("\x00", "")


class PostgresVectorStore(VectorStorePort):
    """Vector store adapter using PostgreSQL with pgvector extension."""

    def __init__(
        self,
        database_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        command_timeout: float = 30.0,
        acquire_timeout: float = 10.0,
    ):
        """Initialize the PostgreSQL vector store.

        Args:
            database_url: PostgreSQL connection URL.
            pool_min_size: Minimum connection pool size.
            pool_max_size: Maximum connection pool size.
            command_timeout: Max seconds a single statement may run.
            acquire_timeout: Max seconds to wait for a free connection.
        """
        self._database_url = database_url
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._command_timeout = command_timeout
        self._acquire_timeout = acquire_timeout
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool.

        The lock matters: `if self._pool is None` followed by an await is a
        check-then-act race, so concurrent first requests would each create a
        pool and every one but the last would be leaked.
        """
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    self._database_url,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                    init=self._init_connection,
                    # Disable statement cache for pgbouncer compatibility (Supabase)
                    statement_cache_size=0,
                    command_timeout=self._command_timeout,
                    timeout=self._acquire_timeout,
                )
                logger.info("PostgreSQL connection pool created")
        return self._pool

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize a connection with pgvector support."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their corresponding embeddings.

        The paper is keyed by its arxiv_id (a UNIQUE column), not by the
        per-fetch Paper.id the source mints, so re-ingesting a paper updates
        the existing row and reuses its id instead of colliding on the
        constraint. The paper upsert, the removal of any prior chunks, and the
        new chunk inserts all run in one transaction, so a failure mid-write
        cannot leave an orphan paper row or a half-replaced chunk set.
        """
        if not chunks:
            return

        pool = await self._get_pool()

        # Paper-level metadata is carried on the first chunk (sanitize all text).
        first_chunk = chunks[0]
        paper_title = _sanitize_text(first_chunk.metadata.get("paper_title", ""))
        arxiv_id = _sanitize_text(first_chunk.metadata.get("arxiv_id", ""))
        url = _sanitize_text(first_chunk.metadata.get("url", ""))
        pdf_url = _sanitize_text(first_chunk.metadata.get("pdf_url", ""))
        authors = first_chunk.metadata.get("authors", [])
        authors = [_sanitize_text(a) for a in authors] if isinstance(authors, list) else []
        abstract = _sanitize_text(first_chunk.metadata.get("abstract", ""))

        async with pool.acquire() as conn, conn.transaction():
            # Reuse the existing row's id when this arxiv_id was already
            # ingested; the RETURNING id is the canonical paper id for its chunks.
            paper_id = await conn.fetchval(
                """
                INSERT INTO papers (id, arxiv_id, title, authors, abstract, url, pdf_url)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    url = EXCLUDED.url,
                    pdf_url = EXCLUDED.pdf_url
                RETURNING id
                """,
                first_chunk.paper_id,
                arxiv_id,
                paper_title,
                authors,
                abstract,
                url,
                pdf_url,
            )

            # Replace the paper's chunks wholesale so a re-chunk into fewer
            # pieces leaves no stale tail rows behind.
            await conn.execute("DELETE FROM chunks WHERE paper_id = $1", paper_id)

            # Insert chunks with embeddings (sanitize text fields to remove null bytes)
            await conn.executemany(
                """
                INSERT INTO chunks (id, paper_id, content, chunk_index, section, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                [
                    (
                        chunk.id,
                        paper_id,
                        _sanitize_text(chunk.content),
                        chunk.chunk_index,
                        _sanitize_text(chunk.section),
                        json.dumps(
                            {
                                k: _sanitize_text(v) if isinstance(v, str) else v
                                for k, v in chunk.metadata.items()
                                if k
                                not in (
                                    "paper_title",
                                    "arxiv_id",
                                    "url",
                                    "pdf_url",
                                    "authors",
                                    "abstract",
                                )
                            }
                        ),
                        np.array(embedding, dtype=np.float32),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

        logger.debug(f"Added {len(chunks)} chunks for paper {paper_id}")

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        paper_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding vector."""
        pool = await self._get_pool()

        embedding_vector = np.array(query_embedding, dtype=np.float32)

        async with pool.acquire() as conn:
            if paper_ids:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.id, c.paper_id, c.content, c.chunk_index, c.section, c.metadata,
                        p.title AS paper_title,
                        1 - (c.embedding <=> $1) AS similarity
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    WHERE c.embedding IS NOT NULL AND c.paper_id = ANY($2::uuid[])
                    ORDER BY c.embedding <=> $1
                    LIMIT $3
                    """,
                    embedding_vector,
                    paper_ids,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.id, c.paper_id, c.content, c.chunk_index, c.section, c.metadata,
                        p.title AS paper_title,
                        1 - (c.embedding <=> $1) AS similarity
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> $1
                    LIMIT $2
                    """,
                    embedding_vector,
                    top_k,
                )

        chunks_with_scores: list[tuple[Chunk, float]] = []
        for row in rows:
            # Parse metadata from JSON if it's a string
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}
            elif metadata is None:
                metadata = {}

            chunk = Chunk(
                id=str(row["id"]),
                paper_id=str(row["paper_id"]),
                content=row["content"],
                chunk_index=row["chunk_index"],
                section=row["section"],
                metadata={
                    **metadata,
                    "paper_title": row["paper_title"],
                },
            )
            chunks_with_scores.append((chunk, float(row["similarity"])))

        return chunks_with_scores

    async def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            paper_count = await conn.fetchval("SELECT COUNT(*) FROM papers")

        return {
            "chunk_count": chunk_count or 0,
            "paper_count": paper_count or 0,
        }

    async def list_papers(self) -> list[dict]:
        """List all papers that have chunks in the store."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    p.id AS paper_id,
                    p.arxiv_id,
                    p.title,
                    p.authors,
                    p.abstract,
                    p.url,
                    p.pdf_url,
                    p.ingested_at,
                    COUNT(c.id) AS chunk_count
                FROM papers p
                LEFT JOIN chunks c ON c.paper_id = p.id
                GROUP BY p.id
                ORDER BY p.ingested_at DESC
                """
            )

        return [
            {
                "paper_id": str(row["paper_id"]),
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "authors": row["authors"] or [],
                "abstract": row["abstract"] or "",
                "url": row["url"],
                "pdf_url": row["pdf_url"],
                "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    async def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a given paper."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Get count before delete
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE paper_id = $1",
                paper_id,
            )

            # Delete paper (chunks will be cascade deleted)
            await conn.execute("DELETE FROM papers WHERE id = $1", paper_id)

        logger.debug(f"Deleted paper {paper_id} with {count} chunks")
        return count or 0

    async def get_paper_embeddings(self) -> list[tuple[str, list[float]]]:
        """Get mean embedding for each paper.

        Computes the mean of all chunk embeddings for each paper.

        Returns:
            List of (paper_id, mean_embedding) tuples.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT paper_id, embedding
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY paper_id
                """
            )

        if not rows:
            return []

        # Group embeddings by paper and compute mean
        paper_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            paper_id = str(row["paper_id"])
            embedding = row["embedding"]
            if embedding is not None:
                # pgvector >= 0.4 decodes columns as Vector objects instead of
                # ndarray-compatible sequences
                if hasattr(embedding, "to_numpy"):
                    embedding = embedding.to_numpy()
                paper_embeddings[paper_id].append(np.asarray(embedding, dtype=np.float32))

        return [
            (paper_id, np.mean(embeddings, axis=0).tolist())
            for paper_id, embeddings in paper_embeddings.items()
        ]

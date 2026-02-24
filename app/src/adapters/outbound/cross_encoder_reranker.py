"""Cross-encoder reranker adapter using sentence-transformers."""

import asyncio
import os
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.domain.entities.chunk import Chunk
from src.domain.ports.reranker import RerankerPort


class CrossEncoderReranker(RerankerPort):
    """Reranker adapter using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        local_files_only: bool = False,
    ):
        """Initialize the reranker adapter.

        Args:
            model_name: Name of the cross-encoder model to use.
            local_files_only: If True, only use locally cached models (no network).
        """
        self._model_name = model_name
        self._local_files_only = local_files_only or os.getenv("HF_HUB_OFFLINE", "0") == "1"
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = _get_model(self._model_name, self._local_files_only)
        return self._model

    def preload(self) -> None:
        """Preload the model (call at startup to avoid cold start on first query)."""
        _ = self.model

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks by relevance to the query.

        Args:
            query: The user's question.
            chunks: Chunks to rerank.
            top_k: Optional limit on returned chunks. If None, return all.

        Returns:
            List of (chunk, score) tuples sorted by relevance descending.
        """
        if not chunks:
            return []

        # Build query-document pairs for the cross-encoder
        pairs = [(query, chunk.content) for chunk in chunks]

        # Get scores from cross-encoder (sync operation wrapped in thread)
        scores = await asyncio.to_thread(self.model.predict, pairs, show_progress_bar=False)

        # Pair chunks with scores and sort by score descending
        chunk_scores = list(zip(chunks, scores.tolist(), strict=True))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            chunk_scores = chunk_scores[:top_k]

        return chunk_scores


@lru_cache(maxsize=2)
def _get_model(model_name: str, local_files_only: bool = False) -> CrossEncoder:
    """Cache models to avoid reloading."""
    return CrossEncoder(model_name, local_files_only=local_files_only)

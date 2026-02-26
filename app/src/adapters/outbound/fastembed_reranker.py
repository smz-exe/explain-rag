"""Cross-encoder reranker adapter using FastEmbed (ONNX-based)."""

import asyncio
import os

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.domain.entities.chunk import Chunk
from src.domain.ports.reranker import RerankerPort


class FastEmbedReranker(RerankerPort):
    """Reranker adapter using FastEmbed's TextCrossEncoder with ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | None = None,
    ):
        """Initialize the reranker adapter.

        Args:
            model_name: Name of the cross-encoder model to use.
            cache_dir: Directory to cache downloaded models.
        """
        self._model_name = model_name
        self._cache_dir = cache_dir or os.getenv("HF_HOME", None)
        self._model: TextCrossEncoder | None = None

    @property
    def model(self) -> TextCrossEncoder:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = TextCrossEncoder(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
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

        documents = [chunk.content for chunk in chunks]

        def _rerank() -> list[float]:
            scores = list(self.model.rerank(query, documents))
            return scores

        scores = await asyncio.to_thread(_rerank)

        # Pair chunks with scores and sort by score descending
        chunk_scores = list(zip(chunks, scores, strict=True))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            chunk_scores = chunk_scores[:top_k]

        return chunk_scores

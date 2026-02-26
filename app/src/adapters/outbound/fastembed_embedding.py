"""Embedding adapter using FastEmbed (ONNX-based, lightweight alternative to sentence-transformers)."""

import asyncio
import os

from fastembed import TextEmbedding

from src.domain.ports.embedding import EmbeddingPort


class FastEmbedEmbedding(EmbeddingPort):
    """Embedding adapter using FastEmbed with ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
    ):
        """Initialize the embedding adapter.

        Args:
            model_name: Name of the FastEmbed model to use.
            cache_dir: Directory to cache downloaded models.
        """
        self._model_name = model_name
        self._cache_dir = cache_dir or os.getenv("HF_HOME", None)
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
        return self._model

    def preload(self) -> None:
        """Preload the model (call at startup to avoid cold start on first query)."""
        _ = self.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Uses asyncio.to_thread to avoid blocking the event loop.
        """

        def _embed() -> list[list[float]]:
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]

        return await asyncio.to_thread(_embed)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

import asyncio
import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.domain.ports.embedding import EmbeddingPort


class SentenceTransformerEmbedding(EmbeddingPort):
    """Embedding adapter using Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_files_only: bool = False):
        """Initialize the embedding adapter.

        Args:
            model_name: Name of the Sentence-Transformers model to use.
            local_files_only: If True, only use locally cached models (no network).
        """
        self._model_name = model_name
        self._local_files_only = local_files_only or os.getenv("HF_HUB_OFFLINE", "0") == "1"
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = _get_model(self._model_name, self._local_files_only)
        return self._model

    def preload(self) -> None:
        """Preload the model (call at startup to avoid cold start on first query)."""
        _ = self.model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        embeddings = await asyncio.to_thread(
            self.model.encode, texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]


@lru_cache(maxsize=2)
def _get_model(model_name: str, local_files_only: bool = False) -> SentenceTransformer:
    """Cache models to avoid reloading."""
    return SentenceTransformer(model_name, local_files_only=local_files_only)

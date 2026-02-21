import asyncio

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.domain.entities.chunk import Chunk
from src.domain.ports.vector_store import VectorStorePort


class ChromaVectorStore(VectorStorePort):
    """Vector store adapter using ChromaDB."""

    COLLECTION_NAME = "explainrag_chunks"

    def __init__(self, persist_dir: str = "./data/chroma"):
        """Initialize the ChromaDB vector store.

        Args:
            persist_dir: Directory for persistent storage.
        """
        self._persist_dir = persist_dir
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy-load the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the chunks collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their corresponding embeddings."""
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section or "",
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]

        await asyncio.to_thread(
            self.collection.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding vector."""
        where = None
        if filter:
            where = filter

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks_with_scores: list[tuple[Chunk, float]] = []

        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert cosine distance to similarity score
                similarity = 1.0 - distance

                chunk = Chunk(
                    id=chunk_id,
                    paper_id=metadata.get("paper_id", ""),
                    content=document,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    section=metadata.get("section") or None,
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ("paper_id", "chunk_index", "section")
                    },
                )
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    async def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = await asyncio.to_thread(self.collection.count)

        # Get unique paper count
        all_metadata = await asyncio.to_thread(
            self.collection.get, include=["metadatas"], limit=count if count > 0 else 1
        )
        paper_ids = set()
        if all_metadata["metadatas"]:
            for meta in all_metadata["metadatas"]:
                if meta and "paper_id" in meta:
                    paper_ids.add(meta["paper_id"])

        return {
            "chunk_count": count,
            "paper_count": len(paper_ids),
        }

    async def list_papers(self) -> list[dict]:
        """List all papers that have chunks in the store."""
        count = await asyncio.to_thread(self.collection.count)
        if count == 0:
            return []

        all_data = await asyncio.to_thread(self.collection.get, include=["metadatas"], limit=count)

        papers: dict[str, dict] = {}
        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if meta and "paper_id" in meta:
                    paper_id = meta["paper_id"]
                    if paper_id not in papers:
                        papers[paper_id] = {
                            "paper_id": paper_id,
                            "arxiv_id": meta.get("arxiv_id", ""),
                            "title": meta.get("paper_title", ""),
                            "chunk_count": 0,
                        }
                    papers[paper_id]["chunk_count"] += 1

        return list(papers.values())

    async def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a given paper."""
        # First, get count of chunks to delete
        results = await asyncio.to_thread(
            self.collection.get, where={"paper_id": paper_id}, include=[]
        )
        count = len(results["ids"]) if results["ids"] else 0

        if count > 0:
            await asyncio.to_thread(self.collection.delete, where={"paper_id": paper_id})

        return count

import logging
from dataclasses import dataclass

from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.paper_source import PaperNotFoundError, PaperSourcePort, PDFParsingError
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a single paper."""

    arxiv_id: str
    title: str
    chunk_count: int
    status: str  # "success" | "error"
    error: str | None = None


@dataclass
class BatchIngestionResult:
    """Result of ingesting multiple papers."""

    ingested: list[IngestionResult]
    errors: list[IngestionResult]


class IngestionService:
    """Service for ingesting papers into the vector store."""

    def __init__(
        self,
        paper_source: PaperSourcePort,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the ingestion service.

        Args:
            paper_source: Adapter for fetching papers.
            embedding: Adapter for generating embeddings.
            vector_store: Adapter for storing chunks.
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self._paper_source = paper_source
        self._embedding = embedding
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_paper(self, arxiv_id: str) -> IngestionResult:
        """Ingest a single paper by arXiv ID.

        Args:
            arxiv_id: The arXiv identifier.

        Returns:
            IngestionResult with status and details.
        """
        try:
            # Fetch paper metadata
            logger.info(f"Fetching paper: {arxiv_id}")
            paper = await self._paper_source.fetch_by_id(arxiv_id)

            # Extract chunks from PDF
            logger.info(f"Extracting chunks from: {paper.title}")
            chunks = await self._paper_source.extract_chunks(
                paper, self._chunk_size, self._chunk_overlap
            )

            if not chunks:
                return IngestionResult(
                    arxiv_id=arxiv_id,
                    title=paper.title,
                    chunk_count=0,
                    status="error",
                    error="No chunks extracted from PDF",
                )

            # Add paper metadata to chunks
            for chunk in chunks:
                chunk.metadata["arxiv_id"] = paper.arxiv_id
                chunk.metadata["paper_title"] = paper.title

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            texts = [chunk.content for chunk in chunks]
            embeddings = await self._embedding.embed_texts(texts)

            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector store")
            await self._vector_store.add_chunks(chunks, embeddings)

            # Update paper chunk count
            paper.chunk_count = len(chunks)

            return IngestionResult(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                chunk_count=len(chunks),
                status="success",
            )

        except PaperNotFoundError as e:
            logger.error(f"Paper not found: {arxiv_id}")
            return IngestionResult(
                arxiv_id=arxiv_id,
                title="",
                chunk_count=0,
                status="error",
                error=str(e),
            )

        except PDFParsingError as e:
            logger.error(f"PDF parsing error for {arxiv_id}: {e}")
            return IngestionResult(
                arxiv_id=arxiv_id,
                title="",
                chunk_count=0,
                status="error",
                error=str(e),
            )

        except Exception as e:
            logger.exception(f"Unexpected error ingesting {arxiv_id}")
            return IngestionResult(
                arxiv_id=arxiv_id,
                title="",
                chunk_count=0,
                status="error",
                error=str(e),
            )

    async def ingest_papers(self, arxiv_ids: list[str]) -> BatchIngestionResult:
        """Ingest multiple papers by arXiv IDs.

        Args:
            arxiv_ids: List of arXiv identifiers.

        Returns:
            BatchIngestionResult with successful and failed ingestions.
        """
        ingested = []
        errors = []

        for arxiv_id in arxiv_ids:
            result = await self.ingest_paper(arxiv_id)
            if result.status == "success":
                ingested.append(result)
            else:
                errors.append(result)

        return BatchIngestionResult(ingested=ingested, errors=errors)

    async def search_and_ingest(self, query: str, max_results: int = 5) -> BatchIngestionResult:
        """Search for papers and ingest the results.

        Args:
            query: Search query string.
            max_results: Maximum number of papers to ingest.

        Returns:
            BatchIngestionResult with successful and failed ingestions.
        """
        papers = await self._paper_source.search(query, max_results)
        arxiv_ids = [paper.arxiv_id for paper in papers]
        return await self.ingest_papers(arxiv_ids)

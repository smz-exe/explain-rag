import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.inbound.http import health, ingest, papers
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.adapters.outbound.chroma_store import ChromaVectorStore
from src.adapters.outbound.st_embedding import SentenceTransformerEmbedding
from src.application.ingestion_service import IngestionService
from src.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()

    # Initialize outbound adapters
    logger.info(f"Initializing embedding adapter: {settings.embedding_model}")
    embedding = SentenceTransformerEmbedding(model_name=settings.embedding_model)

    logger.info(f"Initializing vector store: {settings.chroma_persist_dir}")
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)

    logger.info("Initializing arXiv paper source")
    paper_source = ArxivPaperSource()

    # Initialize application services
    ingestion_service = IngestionService(
        paper_source=paper_source,
        embedding=embedding,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Create FastAPI app
    app = FastAPI(
        title="ExplainRAG",
        description="Explainable Retrieval-Augmented Generation for academic papers",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(ingest.create_router(ingestion_service))
    app.include_router(papers.create_router(vector_store))
    app.include_router(health.create_router(vector_store))

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "ExplainRAG",
            "version": "0.1.0",
            "docs": "/docs",
        }

    logger.info("ExplainRAG application initialized")
    return app


app = create_app()

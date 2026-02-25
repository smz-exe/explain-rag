import logging
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.inbound.http import (
    auth,
    coordinates,
    evaluation,
    health,
    ingest,
    papers,
    query,
    stats,
)
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.adapters.outbound.chroma_store import ChromaVectorStore
from src.adapters.outbound.cross_encoder_reranker import CrossEncoderReranker
from src.adapters.outbound.env_user_storage import EnvUserStorage
from src.adapters.outbound.hdbscan_clusterer import HDBSCANClusterer
from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness
from src.adapters.outbound.langchain_rag import LangChainRAG
from src.adapters.outbound.ragas_evaluator import RAGASEvaluator
from src.adapters.outbound.sqlite_coordinates_storage import SQLiteCoordinatesStorage
from src.adapters.outbound.sqlite_query_storage import SQLiteQueryStorage
from src.adapters.outbound.st_embedding import SentenceTransformerEmbedding
from src.adapters.outbound.umap_reducer import UMAPReducer
from src.application.coordinates_service import CoordinatesService
from src.application.ingestion_service import IngestionService
from src.application.query_service import QueryService
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

    # Set HF_TOKEN for HuggingFace libraries (they read from os.environ)
    hf_token = settings.hf_token.get_secret_value()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Initialize outbound adapters
    logger.info(f"Initializing embedding adapter: {settings.embedding_model}")
    embedding = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        local_files_only=settings.hf_offline_mode,
    )

    logger.info(f"Initializing vector store: {settings.chroma_persist_dir}")
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)

    logger.info("Initializing arXiv paper source")
    paper_source = ArxivPaperSource()

    # Initialize LLM adapters
    api_key = settings.anthropic_api_key.get_secret_value()
    logger.info(f"Initializing LLM adapter: {settings.claude_model}")
    llm = LangChainRAG(
        model=settings.claude_model,
        api_key=api_key,
        max_tokens=settings.claude_max_tokens,
    )

    logger.info("Initializing faithfulness adapter")
    faithfulness = LangChainFaithfulness(
        model=settings.claude_model,
        api_key=api_key,
    )

    logger.info(f"Initializing reranker: {settings.reranker_model}")
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        local_files_only=settings.hf_offline_mode,
    )

    logger.info(f"Initializing query storage: {settings.sqlite_db_path}")
    query_storage = SQLiteQueryStorage(db_path=settings.sqlite_db_path)

    logger.info(f"Initializing coordinates storage: {settings.sqlite_db_path}")
    coordinates_storage = SQLiteCoordinatesStorage(db_path=settings.sqlite_db_path)

    logger.info("Initializing user storage for auth")
    user_storage = EnvUserStorage(
        admin_username=settings.admin_username,
        admin_password_hash=settings.admin_password_hash.get_secret_value(),
    )

    logger.info("Initializing RAGAS evaluator")
    evaluator = RAGASEvaluator(
        model=settings.claude_model,
        api_key=api_key,
        embedding_model=settings.embedding_model,
        max_tokens=settings.claude_max_tokens,
        timeout=settings.claude_timeout,
        max_retries=settings.claude_max_retries,
    )

    logger.info("Initializing dimensionality reduction adapter (UMAP)")
    dim_reducer = UMAPReducer(
        n_neighbors=settings.umap_n_neighbors,
        min_dist=settings.umap_min_dist,
        random_state=42,
    )

    logger.info("Initializing clustering adapter (HDBSCAN)")
    clusterer = HDBSCANClusterer(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
    )

    # Preload models at startup to avoid cold start on first query
    if settings.preload_models:
        logger.info("Preloading models...")
        embedding.preload()
        reranker.preload()
        logger.info("Models preloaded successfully")

    # Initialize application services
    ingestion_service = IngestionService(
        paper_source=paper_source,
        embedding=embedding,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    query_service = QueryService(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        faithfulness=faithfulness,
        reranker=reranker,
        query_storage=query_storage,
        default_top_k=settings.default_top_k,
    )

    coordinates_service = CoordinatesService(
        vector_store=vector_store,
        dim_reducer=dim_reducer,
        clusterer=clusterer,
        storage=coordinates_storage,
    )

    # Define lifespan context manager for startup/shutdown events
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events."""
        # Startup
        logger.info("Running startup tasks...")
        await coordinates_service.initialize()
        logger.info("Startup tasks completed")
        yield
        # Shutdown (if needed in the future)

    # Create FastAPI app
    app = FastAPI(
        title="ExplainRAG",
        description="Explainable Retrieval-Augmented Generation for academic papers",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler for unhandled errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions with a structured error response."""
        logger.error(
            f"Unhandled exception: {exc}\n"
            f"Path: {request.url.path}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc),
                "path": request.url.path,
            },
        )

    # Mount routers
    app.include_router(auth.create_router(user_storage, settings))
    app.include_router(ingest.create_router(ingestion_service))
    app.include_router(papers.create_router(vector_store, paper_source))
    app.include_router(coordinates.create_router(coordinates_service))
    app.include_router(coordinates.create_admin_router(coordinates_service))
    app.include_router(health.create_router(vector_store))
    app.include_router(query.create_router(query_service))
    app.include_router(stats.create_router(vector_store, query_storage))
    app.include_router(evaluation.create_router(evaluator, query_storage))

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

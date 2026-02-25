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
from src.adapters.outbound.postgres_query_storage import PostgresQueryStorage
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.adapters.outbound.ragas_evaluator import RAGASEvaluator
from src.adapters.outbound.sqlite_coordinates_storage import SQLiteCoordinatesStorage
from src.adapters.outbound.sqlite_query_storage import SQLiteQueryStorage
from src.adapters.outbound.st_embedding import SentenceTransformerEmbedding
from src.adapters.outbound.umap_reducer import UMAPReducer
from src.application.coordinates_service import CoordinatesService
from src.application.ingestion_service import IngestionService
from src.application.query_service import QueryService
from src.config import Settings
from src.domain.ports.clustering import ClusteringPort
from src.domain.ports.coordinates_storage import CoordinatesStoragePort
from src.domain.ports.dimensionality_reduction import DimensionalityReductionPort
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.evaluation import EvaluationPort
from src.domain.ports.faithfulness import FaithfulnessPort
from src.domain.ports.llm import LLMPort
from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.reranker import RerankerPort
from src.domain.ports.vector_store import VectorStorePort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    *,
    embedding: EmbeddingPort | None = None,
    vector_store: VectorStorePort | None = None,
    llm: LLMPort | None = None,
    faithfulness: FaithfulnessPort | None = None,
    reranker: RerankerPort | None = None,
    query_storage: QueryStoragePort | None = None,
    coordinates_storage: CoordinatesStoragePort | None = None,
    evaluator: EvaluationPort | None = None,
    dim_reducer: DimensionalityReductionPort | None = None,
    clusterer: ClusteringPort | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    All adapter parameters are optional. If not provided, real production
    adapters will be created. Pass mock adapters for testing.

    Args:
        embedding: Embedding adapter (default: SentenceTransformerEmbedding)
        vector_store: Vector store adapter (default: ChromaVectorStore)
        llm: LLM adapter (default: LangChainRAG)
        faithfulness: Faithfulness adapter (default: LangChainFaithfulness)
        reranker: Reranker adapter (default: CrossEncoderReranker)
        query_storage: Query storage adapter (default: SQLiteQueryStorage)
        coordinates_storage: Coordinates storage adapter (default: SQLiteCoordinatesStorage)
        evaluator: Evaluation adapter (default: RAGASEvaluator)
        dim_reducer: Dimensionality reduction adapter (default: UMAPReducer)
        clusterer: Clustering adapter (default: HDBSCANClusterer)

    Returns:
        Configured FastAPI application
    """
    settings = Settings()

    # Set HF_TOKEN for HuggingFace libraries (they read from os.environ)
    hf_token = settings.hf_token.get_secret_value()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Initialize outbound adapters (use provided or create real ones)
    if embedding is None:
        logger.info(f"Initializing embedding adapter: {settings.embedding_model}")
        embedding = SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            local_files_only=settings.hf_offline_mode,
        )

    if vector_store is None:
        if settings.database_url:
            logger.info("Initializing PostgreSQL vector store")
            vector_store = PostgresVectorStore(
                database_url=settings.database_url,
                pool_min_size=settings.database_pool_min,
                pool_max_size=settings.database_pool_max,
            )
        else:
            logger.info(f"Initializing ChromaDB vector store: {settings.chroma_persist_dir}")
            vector_store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)

    logger.info("Initializing arXiv paper source")
    paper_source = ArxivPaperSource()

    # Initialize LLM adapters
    api_key = settings.anthropic_api_key.get_secret_value()

    if llm is None:
        logger.info(f"Initializing LLM adapter: {settings.claude_model}")
        llm = LangChainRAG(
            model=settings.claude_model,
            api_key=api_key,
            max_tokens=settings.claude_max_tokens,
        )

    if faithfulness is None:
        logger.info("Initializing faithfulness adapter")
        faithfulness = LangChainFaithfulness(
            model=settings.claude_model,
            api_key=api_key,
        )

    if reranker is None:
        logger.info(f"Initializing reranker: {settings.reranker_model}")
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            local_files_only=settings.hf_offline_mode,
        )

    if query_storage is None:
        if settings.database_url:
            logger.info("Initializing PostgreSQL query storage")
            query_storage = PostgresQueryStorage(
                database_url=settings.database_url,
                pool_min_size=settings.database_pool_min,
                pool_max_size=settings.database_pool_max,
            )
        else:
            logger.info(f"Initializing SQLite query storage: {settings.sqlite_db_path}")
            query_storage = SQLiteQueryStorage(db_path=settings.sqlite_db_path)

    if coordinates_storage is None:
        logger.info(f"Initializing coordinates storage: {settings.sqlite_db_path}")
        coordinates_storage = SQLiteCoordinatesStorage(db_path=settings.sqlite_db_path)

    logger.info("Initializing user storage for auth")
    user_storage = EnvUserStorage(
        admin_username=settings.admin_username,
        admin_password_hash=settings.admin_password_hash.get_secret_value(),
    )

    if evaluator is None:
        logger.info("Initializing RAGAS evaluator")
        evaluator = RAGASEvaluator(
            model=settings.claude_model,
            api_key=api_key,
            embedding_model=settings.embedding_model,
            max_tokens=settings.claude_max_tokens,
            timeout=settings.claude_timeout,
            max_retries=settings.claude_max_retries,
        )

    if dim_reducer is None:
        logger.info("Initializing dimensionality reduction adapter (UMAP)")
        dim_reducer = UMAPReducer(
            n_neighbors=settings.umap_n_neighbors,
            min_dist=settings.umap_min_dist,
            random_state=42,
        )

    if clusterer is None:
        logger.info("Initializing clustering adapter (HDBSCAN)")
        clusterer = HDBSCANClusterer(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
        )

    # Preload models at startup to avoid cold start on first query
    # Only preload if using real adapters (they have preload method)
    if settings.preload_models:
        if hasattr(embedding, "preload"):
            logger.info("Preloading embedding model...")
            embedding.preload()
        if hasattr(reranker, "preload"):
            logger.info("Preloading reranker model...")
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
        # Shutdown - close connection pools
        logger.info("Running shutdown tasks...")
        if hasattr(vector_store, "close"):
            await vector_store.close()
        if hasattr(query_storage, "close"):
            await query_storage.close()
        logger.info("Shutdown tasks completed")

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

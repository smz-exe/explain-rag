import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.inbound.http import auth, health, ingest, papers, query, stats
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.adapters.outbound.chroma_store import ChromaVectorStore
from src.adapters.outbound.cross_encoder_reranker import CrossEncoderReranker
from src.adapters.outbound.env_user_storage import EnvUserStorage
from src.adapters.outbound.langchain_faithfulness import LangChainFaithfulness
from src.adapters.outbound.langchain_rag import LangChainRAG
from src.adapters.outbound.sqlite_query_storage import SQLiteQueryStorage
from src.adapters.outbound.st_embedding import SentenceTransformerEmbedding
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
    logger.info(f"Initializing LLM adapter: {settings.claude_model}")
    llm = LangChainRAG(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.claude_max_tokens,
    )

    logger.info("Initializing faithfulness adapter")
    faithfulness = LangChainFaithfulness(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
    )

    logger.info(f"Initializing reranker: {settings.reranker_model}")
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        local_files_only=settings.hf_offline_mode,
    )

    logger.info(f"Initializing query storage: {settings.sqlite_db_path}")
    query_storage = SQLiteQueryStorage(db_path=settings.sqlite_db_path)

    logger.info("Initializing user storage for auth")
    user_storage = EnvUserStorage(
        admin_username=settings.admin_username,
        admin_password_hash=settings.admin_password_hash,
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
    app.include_router(papers.create_router(vector_store))
    app.include_router(health.create_router(vector_store))
    app.include_router(query.create_router(query_service))
    app.include_router(stats.create_router(vector_store, query_storage))

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

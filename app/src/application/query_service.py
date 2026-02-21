import logging
import time
import uuid

from src.domain.entities.explanation import ExplanationTrace, FaithfulnessResult
from src.domain.entities.query import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.faithfulness import FaithfulnessPort
from src.domain.ports.llm import InsufficientContextError, LLMPort
from src.domain.ports.query_storage import QueryNotFoundError, QueryStoragePort
from src.domain.ports.reranker import RerankerPort
from src.domain.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


class QueryService:
    """Service orchestrating the full query pipeline."""

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        faithfulness: FaithfulnessPort,
        reranker: RerankerPort | None = None,
        query_storage: QueryStoragePort | None = None,
        default_top_k: int = 10,
    ):
        """Initialize the query service.

        Args:
            embedding: Adapter for generating embeddings.
            vector_store: Adapter for vector search.
            llm: Adapter for answer generation.
            faithfulness: Adapter for faithfulness verification.
            reranker: Optional adapter for cross-encoder reranking.
            query_storage: Adapter for persistent query storage.
            default_top_k: Default number of chunks to retrieve.
        """
        self._embedding = embedding
        self._vector_store = vector_store
        self._llm = llm
        self._faithfulness = faithfulness
        self._reranker = reranker
        self._query_storage = query_storage
        self._default_top_k = default_top_k

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute the full query pipeline.

        Pipeline steps:
        1. Embed query
        2. Search vector store
        3. Generate answer with LLM
        4. Verify faithfulness
        5. Build response with timing trace

        Args:
            request: The query request.

        Returns:
            Complete QueryResponse with all explainability data.
        """
        query_id = str(uuid.uuid4())
        top_k = request.top_k or self._default_top_k

        logger.info(f"Processing query {query_id}: {request.question[:50]}...")

        total_start = time.perf_counter()

        # Step 1: Embed query
        logger.debug("Step 1: Embedding query")
        embed_start = time.perf_counter()
        query_embedding = await self._embedding.embed_query(request.question)
        embed_time = (time.perf_counter() - embed_start) * 1000

        # Step 2: Search vector store
        logger.debug(f"Step 2: Searching for top-{top_k} chunks")
        search_start = time.perf_counter()

        # Build filter if paper_ids specified
        search_filter = None
        if request.paper_ids:
            search_filter = {"paper_id": {"$in": request.paper_ids}}

        search_results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=search_filter,
        )
        retrieval_time = (time.perf_counter() - search_start) * 1000

        # Check if we have results
        if not search_results:
            logger.warning(f"No chunks found for query: {request.question}")
            return await self._build_insufficient_context_response(
                query_id=query_id,
                question=request.question,
                embed_time=embed_time,
                retrieval_time=retrieval_time,
            )

        # Step 3: Optional reranking
        reranking_time: float | None = None
        rerank_scores: dict[str, float] = {}

        if request.enable_reranking and self._reranker is not None:
            logger.debug("Step 3: Reranking chunks")
            rerank_start = time.perf_counter()

            # Extract chunks from search results for reranking
            chunks_to_rerank = [chunk for chunk, _ in search_results]
            reranked = await self._reranker.rerank(
                query=request.question,
                chunks=chunks_to_rerank,
                top_k=top_k,
            )

            # Build rerank scores map and update search_results order
            rerank_scores = {chunk.id: score for chunk, score in reranked}
            # Preserve original similarity scores while using reranked order
            original_scores = {chunk.id: score for chunk, score in search_results}
            search_results = [
                (chunk, original_scores[chunk.id]) for chunk, _ in reranked
            ]

            reranking_time = (time.perf_counter() - rerank_start) * 1000
            logger.debug(f"Reranking completed in {reranking_time:.1f}ms")

        # Step 4: Prepare chunks and build RetrievedChunk objects
        chunks = []
        retrieved_chunks = []
        for rank, (chunk, score) in enumerate(search_results, start=1):
            chunks.append(chunk)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    paper_id=chunk.paper_id,
                    paper_title=chunk.metadata.get("paper_title", ""),
                    content=chunk.content,
                    similarity_score=score,
                    rerank_score=rerank_scores.get(chunk.id),
                    rank=rank,
                )
            )

        # Step 4: Generate answer with LLM
        logger.debug("Step 3: Generating answer")
        gen_start = time.perf_counter()
        try:
            generation_result = await self._llm.generate(
                question=request.question,
                chunks=chunks,
            )
            answer = generation_result.answer
            citations = generation_result.citations
        except InsufficientContextError as e:
            logger.warning(f"Insufficient context: {e}")
            gen_time = (time.perf_counter() - gen_start) * 1000
            return await self._build_insufficient_context_response(
                query_id=query_id,
                question=request.question,
                embed_time=embed_time,
                retrieval_time=retrieval_time,
                generation_time=gen_time,
                retrieved_chunks=retrieved_chunks,
            )
        gen_time = (time.perf_counter() - gen_start) * 1000

        # Step 5: Verify faithfulness
        logger.debug("Step 4: Verifying faithfulness")
        faith_start = time.perf_counter()
        faithfulness_result = await self._faithfulness.verify(
            answer=answer,
            chunks=chunks,
        )
        faith_time = (time.perf_counter() - faith_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # Build trace
        trace = ExplanationTrace(
            embedding_time_ms=embed_time,
            retrieval_time_ms=retrieval_time,
            reranking_time_ms=reranking_time,
            generation_time_ms=gen_time,
            faithfulness_time_ms=faith_time,
            total_time_ms=total_time,
        )

        # Build response
        response = QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            faithfulness=faithfulness_result,
            trace=trace,
        )

        # Store for later retrieval
        if self._query_storage:
            await self._query_storage.store(response)

        logger.info(f"Query {query_id} completed in {total_time:.1f}ms")

        return response

    async def get_query(self, query_id: str) -> QueryResponse:
        """Retrieve a stored query response by ID.

        Args:
            query_id: The query UUID.

        Returns:
            The stored QueryResponse.

        Raises:
            QueryNotFoundError: If query ID not found.
        """
        if self._query_storage is None:
            raise QueryNotFoundError(query_id)

        response = await self._query_storage.get(query_id)
        if response is None:
            raise QueryNotFoundError(query_id)
        return response

    async def list_recent_queries(self, limit: int = 20) -> list[dict]:
        """List recent queries with summary information.

        Args:
            limit: Maximum number of queries to return.

        Returns:
            List of query summaries.
        """
        if self._query_storage is None:
            return []
        return await self._query_storage.list_recent(limit)

    async def _build_insufficient_context_response(
        self,
        query_id: str,
        question: str,
        embed_time: float,
        retrieval_time: float,
        generation_time: float = 0.0,
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> QueryResponse:
        """Build a response for insufficient context case."""
        trace = ExplanationTrace(
            embedding_time_ms=embed_time,
            retrieval_time_ms=retrieval_time,
            reranking_time_ms=None,
            generation_time_ms=generation_time,
            faithfulness_time_ms=0.0,
            total_time_ms=embed_time + retrieval_time + generation_time,
        )

        response = QueryResponse(
            query_id=query_id,
            question=question,
            answer="I cannot answer this question based on the available context. "
            "Please try a different question or ingest more relevant papers.",
            citations=[],
            retrieved_chunks=retrieved_chunks or [],
            faithfulness=FaithfulnessResult(score=0.0, claims=[]),
            trace=trace,
        )

        if self._query_storage:
            await self._query_storage.store(response)
        return response

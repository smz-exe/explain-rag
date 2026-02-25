"""RAGAS evaluation adapter using Claude for LLM-based metrics."""

import asyncio
import logging

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

from src.domain.ports.evaluation import EvaluationError, EvaluationMetrics, EvaluationPort

logger = logging.getLogger(__name__)


class RAGASEvaluator(EvaluationPort):
    """RAGAS evaluation adapter using Anthropic Claude."""

    def __init__(
        self,
        model: str,
        api_key: str,
        embedding_model: str,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        """Initialize the RAGAS evaluator.

        Args:
            model: Claude model name (e.g., 'claude-sonnet-4-20250514').
            api_key: Anthropic API key.
            max_tokens: Maximum tokens for LLM responses.
            embedding_model: HuggingFace embedding model name.
            timeout: Timeout in seconds for Claude API calls.
            max_retries: Max retries for transient failures.
        """
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._llm: LangchainLLMWrapper | None = None
        self._embeddings: LangchainEmbeddingsWrapper | None = None

    @property
    def llm(self) -> LangchainLLMWrapper:
        """Lazy-load the LLM wrapper for RAGAS."""
        if self._llm is None:
            anthropic_llm = ChatAnthropic(
                model=self._model,
                api_key=self._api_key,
                max_tokens=self._max_tokens,
                temperature=0.0,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
            self._llm = LangchainLLMWrapper(anthropic_llm)
        return self._llm

    @property
    def embeddings(self) -> LangchainEmbeddingsWrapper:
        """Lazy-load the HuggingFace embeddings wrapped for RAGAS."""
        if self._embeddings is None:
            hf_embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)
            self._embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        return self._embeddings

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationMetrics:
        """Evaluate a RAG response using RAGAS metrics.

        Args:
            question: The original question.
            answer: The generated answer.
            contexts: List of retrieved context chunks.
            ground_truth: Optional ground truth for context_recall.

        Returns:
            EvaluationMetrics with all computed scores.

        Raises:
            EvaluationError: If evaluation fails.
        """
        logger.info(f"Evaluating RAG response for question: {question[:50]}...")

        # Build the evaluation dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        # Add reference if provided (required for context_precision and context_recall)
        if ground_truth:
            data["reference"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics based on available data
        # faithfulness and answer_relevancy don't need ground_truth
        # context_precision and context_recall require ground_truth (reference)
        metrics = [Faithfulness(), AnswerRelevancy()]
        if ground_truth:
            metrics.extend([ContextPrecision(), ContextRecall()])

        def run_evaluation():
            """Synchronous evaluation wrapper for asyncio.to_thread."""
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                show_progress=False,
            )

        try:
            result = await asyncio.to_thread(run_evaluation)

            # Extract scores from result
            df = result.to_pandas()
            scores = df.iloc[0].to_dict()

            return EvaluationMetrics(
                faithfulness=float(scores.get("faithfulness", 0.0)),
                answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
                context_precision=float(scores.get("context_precision", 0.0)),
                context_recall=float(scores.get("context_recall", 0.0)) if ground_truth else 0.0,
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise EvaluationError(f"RAGAS evaluation failed: {e}") from e

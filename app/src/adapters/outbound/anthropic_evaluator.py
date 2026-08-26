"""RAG evaluation adapter calling the Anthropic SDK directly.

Replaces the former RAGAS-based evaluator (see docs/DECISIONS.md): ragas
required a langchain version pinned to Feb 2026 and broke production when
rebuilt. The metric *definitions* differ slightly from RAGAS:

- faithfulness: fraction of answer claims the judge marks as supported by the
  contexts (RAGAS: same idea, different decomposition prompts)
- answer_relevancy: direct judgment full/partial/none mapped to 1.0/0.5/0.0
  (RAGAS: embedding similarity of reverse-engineered questions)
- context_precision: fraction of contexts relevant to the question — computed
  WITHOUT ground truth (RAGAS required a reference and this metric was
  effectively always 0.0 in this app)
- context_recall: fraction of reference-answer statements covered by the
  contexts; still requires ground truth, otherwise 0.0

The judge returns per-item booleans and the ratios are computed in code,
rather than asking the model for bare scores.
"""

import logging

from anthropic import AsyncAnthropic

from src.adapters.outbound.anthropic_client import (
    build_client,
    parse_json_response,
    response_text,
)
from src.domain.ports.evaluation import EvaluationError, EvaluationMetrics, EvaluationPort

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating the quality of a retrieval-augmented answer.

Question:
{question}

Answer:
{answer}

Context chunks:
{contexts}
{reference_section}
Evaluate the answer and contexts. Respond with a JSON object:
{{
    "claims": [
        {{"text": "<one factual claim from the answer>", "supported_by_context": true or false}}
    ],
    "answer_relevance": "full" or "partial" or "none",
    "contexts": [
        {{"index": <1-based chunk number>, "relevant_to_question": true or false}}
    ]{reference_field}
}}

Rules:
- Decompose the answer into its distinct factual claims (empty array if none)
- A claim is supported only if the context chunks state or directly imply it
- "answer_relevance" is "full" if the answer addresses the question directly,
  "partial" if it only partly addresses it, "none" if it does not
- Include one entry in "contexts" for every chunk

Output only the JSON object, no other text:"""

REFERENCE_SECTION = """
Reference answer (ground truth):
{ground_truth}
"""

REFERENCE_FIELD = """,
    "reference_statements": [
        {{"text": "<one statement from the reference answer>", "covered_by_context": true or false}}
    ]"""


class AnthropicEvaluator(EvaluationPort):
    """LLM-judge evaluation adapter using the Anthropic SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: AsyncAnthropic | None = None,
    ):
        """Initialize the evaluator.

        Args:
            model: Anthropic model name.
            api_key: Anthropic API key.
            max_tokens: Maximum tokens for the judge response.
            timeout: Request timeout in seconds.
            max_retries: SDK retry count for transient failures.
            client: Optional pre-built client (used in tests).
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or build_client(api_key, timeout, max_retries)

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationMetrics:
        """Evaluate a RAG response with a single LLM-judge call."""
        logger.info(f"Evaluating RAG response for question: {question[:50]}...")

        contexts_text = "\n".join(
            f"Chunk [{i}]:\n{context}\n" for i, context in enumerate(contexts, start=1)
        )
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            contexts=contexts_text,
            reference_section=REFERENCE_SECTION.format(ground_truth=ground_truth)
            if ground_truth
            else "",
            reference_field=REFERENCE_FIELD if ground_truth else "",
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = parse_json_response(response_text(response))
            # Inside the boundary: a malformed verdict is an evaluation failure,
            # not an unhandled exception surfacing as a 500.
            return self._compute_metrics(verdict, has_ground_truth=ground_truth is not None)
        except EvaluationError:
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation failed: {e}") from e

    def _compute_metrics(self, verdict: dict, *, has_ground_truth: bool) -> EvaluationMetrics:
        """Turn the judge's per-item booleans into metric ratios."""
        if not isinstance(verdict, dict):
            raise EvaluationError(f"Judge returned non-object payload: {verdict!r}")

        raw_claims = verdict.get("claims")
        claims = (
            [c for c in raw_claims if isinstance(c, dict)] if isinstance(raw_claims, list) else []
        )
        if not claims:
            # Scoring this 1.0 made "the judge told us nothing" indistinguishable from
            # "every claim was supported" in the published benchmark means.
            raise EvaluationError(f"Judge returned no usable claims: {verdict!r}")
        faithfulness = sum(1 for c in claims if c.get("supported_by_context") is True) / len(claims)

        relevance_scores = {"full": 1.0, "partial": 0.5, "none": 0.0}
        relevance = verdict.get("answer_relevance")
        answer_relevancy = (
            relevance_scores.get(relevance, 0.0) if isinstance(relevance, str) else 0.0
        )

        contexts = [c for c in verdict.get("contexts", []) if isinstance(c, dict)]
        context_precision = (
            sum(1 for c in contexts if c.get("relevant_to_question") is True) / len(contexts)
            if contexts
            else 0.0
        )

        context_recall = 0.0
        if has_ground_truth:
            statements = [s for s in verdict.get("reference_statements", []) if isinstance(s, dict)]
            if statements:
                context_recall = sum(
                    1 for s in statements if s.get("covered_by_context") is True
                ) / len(statements)

        return EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        )

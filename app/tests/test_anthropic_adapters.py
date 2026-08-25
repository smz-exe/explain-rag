"""Unit tests for the Anthropic SDK adapters (RAG, faithfulness, evaluator)."""

import json
from types import SimpleNamespace

import pytest

from src.adapters.outbound.anthropic_evaluator import AnthropicEvaluator
from src.adapters.outbound.anthropic_faithfulness import AnthropicFaithfulness
from src.adapters.outbound.anthropic_rag import AnthropicRAG
from src.domain.ports.evaluation import EvaluationError
from src.domain.ports.faithfulness import FaithfulnessVerificationError
from src.domain.ports.llm import InsufficientContextError, LLMGenerationError


class FakeMessages:
    """Stands in for AsyncAnthropic().messages, replaying canned text responses."""

    def __init__(self, responses: list[str], error: Exception | None = None):
        self._responses = list(responses)
        self._error = error
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        text = self._responses.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """Minimal stand-in for AsyncAnthropic."""

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.messages = FakeMessages(responses or [], error)


class TestAnthropicRAG:
    """Tests for the RAG generation adapter."""

    def make_adapter(self, responses=None, error=None) -> tuple[AnthropicRAG, FakeClient]:
        client = FakeClient(responses, error)
        return AnthropicRAG(model="claude-sonnet-5", client=client), client

    async def test_generate_returns_answer_with_citations(self, sample_chunks):
        answer = "Self-attention relates positions [1]. It is used widely [2][3]."
        adapter, client = self.make_adapter([answer])

        result = await adapter.generate("What is self-attention?", sample_chunks)

        assert result.answer == answer
        assert len(result.citations) == 2
        assert result.citations[0].chunk_ids == [sample_chunks[0].id]
        assert result.citations[1].chunk_ids == [sample_chunks[1].id, sample_chunks[2].id]

    async def test_generate_sends_system_prompt_and_formatted_chunks(self, sample_chunks):
        adapter, client = self.make_adapter(["Answer [1]."])

        await adapter.generate("What is self-attention?", sample_chunks)

        request = client.messages.requests[0]
        assert request["model"] == "claude-sonnet-5"
        assert "CITATION RULES" in request["system"]
        user_content = request["messages"][0]["content"]
        assert "Chunk [1]" in user_content
        assert sample_chunks[0].content in user_content
        assert "What is self-attention?" in user_content

    async def test_empty_chunks_raises_insufficient_context(self):
        adapter, _ = self.make_adapter([])

        with pytest.raises(InsufficientContextError):
            await adapter.generate("Question?", [])

    async def test_cannot_answer_response_raises_insufficient_context(self, sample_chunks):
        adapter, _ = self.make_adapter(
            ["I cannot answer this question based on the available context."]
        )

        with pytest.raises(InsufficientContextError):
            await adapter.generate("Unanswerable?", sample_chunks)

    async def test_api_error_raises_generation_error(self, sample_chunks):
        adapter, _ = self.make_adapter(error=RuntimeError("boom"))

        with pytest.raises(LLMGenerationError):
            await adapter.generate("Question?", sample_chunks)

    async def test_out_of_range_citations_are_ignored(self, sample_chunks):
        adapter, _ = self.make_adapter(["Claim with a bad reference [9]."])

        result = await adapter.generate("Question?", sample_chunks)

        assert result.citations == []


class TestAnthropicFaithfulness:
    """Tests for the faithfulness verification adapter."""

    def make_adapter(self, responses=None, error=None) -> AnthropicFaithfulness:
        return AnthropicFaithfulness(model="claude-sonnet-5", client=FakeClient(responses, error))

    async def test_verify_scores_mixed_verdicts(self, sample_chunks):
        claims = json.dumps(["Claim A.", "Claim B.", "Claim C."])
        verdicts = json.dumps(
            [
                {"claim_index": 0, "verdict": "supported", "evidence_chunk_indices": [1]},
                {"claim_index": 1, "verdict": "partial", "evidence_chunk_indices": [2]},
                {"claim_index": 2, "verdict": "unsupported", "evidence_chunk_indices": []},
            ]
        )
        adapter = self.make_adapter([claims, verdicts])

        result = await adapter.verify("Answer text.", sample_chunks)

        assert result.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert result.claims[0].evidence_chunk_ids == [sample_chunks[0].id]
        assert result.claims[2].verdict == "unsupported"

    async def test_markdown_fenced_json_is_parsed(self, sample_chunks):
        claims = '```json\n["Claim A."]\n```'
        verdicts = '```json\n[{"claim_index": 0, "verdict": "supported", "evidence_chunk_indices": [1]}]\n```'
        adapter = self.make_adapter([claims, verdicts])

        result = await adapter.verify("Answer.", sample_chunks)

        assert result.score == 1.0

    async def test_invalid_decompose_json_falls_back_to_sentences(self, sample_chunks):
        verdicts = json.dumps(
            [
                {"claim_index": 0, "verdict": "supported", "evidence_chunk_indices": [1]},
                {"claim_index": 1, "verdict": "supported", "evidence_chunk_indices": [1]},
            ]
        )
        adapter = self.make_adapter(["not json at all", verdicts])

        result = await adapter.verify("First sentence. Second sentence.", sample_chunks)

        assert len(result.claims) == 2
        assert result.claims[0].claim == "First sentence."

    async def test_invalid_verdict_json_marks_all_unsupported(self, sample_chunks):
        adapter = self.make_adapter([json.dumps(["Claim A."]), "not json"])

        result = await adapter.verify("Answer.", sample_chunks)

        assert result.score == 0.0
        assert result.claims[0].verdict == "unsupported"

    async def test_no_claims_returns_perfect_score(self, sample_chunks):
        adapter = self.make_adapter(["[]"])

        result = await adapter.verify("", sample_chunks)

        assert result.score == 1.0
        assert result.claims == []

    async def test_api_error_raises_verification_error(self, sample_chunks):
        adapter = self.make_adapter(error=RuntimeError("boom"))

        with pytest.raises(FaithfulnessVerificationError):
            await adapter.verify("Answer.", sample_chunks)


class TestAnthropicEvaluator:
    """Tests for the LLM-judge evaluation adapter."""

    def make_evaluator(self, responses=None, error=None) -> tuple[AnthropicEvaluator, FakeClient]:
        client = FakeClient(responses, error)
        return AnthropicEvaluator(model="claude-sonnet-5", api_key="test", client=client), client

    def verdict_payload(self, **overrides) -> str:
        payload = {
            "claims": [
                {"text": "Claim A", "supported_by_context": True},
                {"text": "Claim B", "supported_by_context": True},
                {"text": "Claim C", "supported_by_context": False},
            ],
            "answer_relevance": "full",
            "contexts": [
                {"index": 1, "relevant_to_question": True},
                {"index": 2, "relevant_to_question": False},
            ],
        }
        payload.update(overrides)
        return json.dumps(payload)

    async def test_metrics_computed_from_judge_booleans(self):
        evaluator, _ = self.make_evaluator([self.verdict_payload()])

        metrics = await evaluator.evaluate("Q?", "Answer.", ["ctx one", "ctx two"])

        assert metrics.faithfulness == pytest.approx(2 / 3)
        assert metrics.answer_relevancy == 1.0
        assert metrics.context_precision == 0.5
        assert metrics.context_recall == 0.0  # no ground truth

    async def test_ground_truth_enables_context_recall(self):
        payload = self.verdict_payload(
            reference_statements=[
                {"text": "Ref A", "covered_by_context": True},
                {"text": "Ref B", "covered_by_context": False},
            ]
        )
        evaluator, client = self.make_evaluator([payload])

        metrics = await evaluator.evaluate("Q?", "Answer.", ["ctx"], ground_truth="Reference.")

        assert metrics.context_recall == 0.5
        prompt = client.messages.requests[0]["messages"][0]["content"]
        assert "Reference answer (ground truth)" in prompt
        assert "reference_statements" in prompt

    async def test_no_claims_means_perfect_faithfulness(self):
        payload = self.verdict_payload(claims=[])
        evaluator, _ = self.make_evaluator([payload])

        metrics = await evaluator.evaluate("Q?", "Answer.", ["ctx"])

        assert metrics.faithfulness == 1.0

    async def test_partial_relevance_maps_to_half(self):
        payload = self.verdict_payload(answer_relevance="partial")
        evaluator, _ = self.make_evaluator([payload])

        metrics = await evaluator.evaluate("Q?", "Answer.", ["ctx"])

        assert metrics.answer_relevancy == 0.5

    async def test_invalid_json_raises_evaluation_error(self):
        evaluator, _ = self.make_evaluator(["not json"])

        with pytest.raises(EvaluationError):
            await evaluator.evaluate("Q?", "Answer.", ["ctx"])

    async def test_api_error_raises_evaluation_error(self):
        evaluator, _ = self.make_evaluator(error=RuntimeError("boom"))

        with pytest.raises(EvaluationError):
            await evaluator.evaluate("Q?", "Answer.", ["ctx"])


class TestFaithfulnessBatching:
    """Large claim sets must be verified in batches to avoid output truncation.

    A 32-claim verification response with per-claim reasoning exceeded
    max_tokens=2048 in production, truncating the JSON array and marking every
    claim unsupported via the parse fallback (0% faithfulness on a faithful
    answer).
    """

    def make_adapter(self, responses) -> tuple[AnthropicFaithfulness, FakeClient]:
        client = FakeClient(responses)
        return AnthropicFaithfulness(model="claude-sonnet-5", client=client), client

    def verdicts_json(self, count: int) -> str:
        return json.dumps(
            [
                {"claim_index": i, "verdict": "supported", "evidence_chunk_indices": [1]}
                for i in range(count)
            ]
        )

    async def test_large_claim_set_is_verified_in_batches(self, sample_chunks):
        claims = [f"Claim {i}." for i in range(30)]
        # 1 decompose call + batches of 12/12/6
        adapter, client = self.make_adapter(
            [
                json.dumps(claims),
                self.verdicts_json(12),
                self.verdicts_json(12),
                self.verdicts_json(6),
            ]
        )

        result = await adapter.verify("Answer.", sample_chunks)

        assert len(client.messages.requests) == 4
        assert len(result.claims) == 30
        assert result.score == 1.0
        # Batch-local claim_index must map back to the global claim order
        assert result.claims[29].claim == "Claim 29."
        assert result.claims[29].verdict == "supported"

    async def test_verify_max_tokens_scales_with_batch_size(self, sample_chunks):
        claims = [f"Claim {i}." for i in range(12)]
        adapter, client = self.make_adapter([json.dumps(claims), self.verdicts_json(12)])

        await adapter.verify("Answer.", sample_chunks)

        verify_request = client.messages.requests[1]
        assert verify_request["max_tokens"] >= 12 * 150

    async def test_failed_batch_only_affects_its_own_claims(self, sample_chunks):
        claims = [f"Claim {i}." for i in range(13)]
        # First batch (12) parses, second batch (1) is truncated garbage
        adapter, _ = self.make_adapter(
            [json.dumps(claims), self.verdicts_json(12), '[{"claim_index": 0, "verd']
        )

        result = await adapter.verify("Answer.", sample_chunks)

        assert len(result.claims) == 13
        assert all(c.verdict == "supported" for c in result.claims[:12])
        assert result.claims[12].verdict == "unsupported"
        assert result.score == pytest.approx(12 / 13)

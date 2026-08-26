"""Conformance tests: every implementation must match its port's signature.

This file replaces a set of tests that instantiated the conftest fakes and
asserted their own hardcoded return values. Those could not fail — they never
executed a line of src/, which was measurable: removing them changed the
coverage report by exactly zero.

What is worth testing at this boundary is the thing that actually broke in
production: the real adapter and the fake drifting apart. QueryService sent a
filter shape PostgresVectorStore could not bind, every test passed because the
mock implemented a different contract, and every paper-scoped query 500'd.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

from src.adapters.outbound.anthropic_evaluator import AnthropicEvaluator
from src.adapters.outbound.anthropic_faithfulness import AnthropicFaithfulness
from src.adapters.outbound.anthropic_rag import AnthropicRAG
from src.adapters.outbound.arxiv_client import ArxivPaperSource
from src.adapters.outbound.env_user_storage import EnvUserStorage
from src.adapters.outbound.fastembed_embedding import FastEmbedEmbedding
from src.adapters.outbound.fastembed_reranker import FastEmbedReranker
from src.adapters.outbound.postgres_query_storage import PostgresQueryStorage
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore
from src.domain.ports.embedding import EmbeddingPort
from src.domain.ports.evaluation import EvaluationPort
from src.domain.ports.faithfulness import FaithfulnessPort
from src.domain.ports.llm import LLMPort
from src.domain.ports.paper_source import PaperSourcePort
from src.domain.ports.query_storage import QueryStoragePort
from src.domain.ports.reranker import RerankerPort
from src.domain.ports.user_storage import UserStoragePort
from src.domain.ports.vector_store import VectorStorePort
from tests.conftest import (
    MockEmbeddingPort,
    MockEvaluationPort,
    MockFaithfulnessPort,
    MockLLMPort,
    MockQueryStoragePort,
    MockRerankerPort,
    MockVectorStorePort,
)

# (port, real adapter, test fake) — the fake must not be more permissive than
# the adapter it stands in for.
PORT_IMPLEMENTATIONS = [
    (VectorStorePort, PostgresVectorStore, MockVectorStorePort),
    (QueryStoragePort, PostgresQueryStorage, MockQueryStoragePort),
    (EmbeddingPort, FastEmbedEmbedding, MockEmbeddingPort),
    (LLMPort, AnthropicRAG, MockLLMPort),
    (FaithfulnessPort, AnthropicFaithfulness, MockFaithfulnessPort),
    (RerankerPort, FastEmbedReranker, MockRerankerPort),
    (EvaluationPort, AnthropicEvaluator, MockEvaluationPort),
]


def _abstract_methods(port: type) -> list[str]:
    return sorted(getattr(port, "__abstractmethods__", set()))


def _parameters(func) -> list[str]:
    """Parameter names of a method, excluding self."""
    return [name for name in inspect.signature(func).parameters if name != "self"]


def _cases():
    for port, adapter, fake in PORT_IMPLEMENTATIONS:
        for method in _abstract_methods(port):
            yield pytest.param(port, adapter, fake, method, id=f"{port.__name__}.{method}")


@pytest.mark.parametrize(("port", "adapter", "fake", "method"), list(_cases()))
def test_implementations_match_the_port_signature(port, adapter, fake, method):
    """A fake that accepts arguments the real adapter rejects hides real bugs."""
    expected = _parameters(getattr(port, method))

    assert _parameters(getattr(adapter, method)) == expected, (
        f"{adapter.__name__}.{method} does not match {port.__name__}"
    )
    assert _parameters(getattr(fake, method)) == expected, (
        f"{fake.__name__}.{method} does not match {port.__name__}"
    )


@pytest.mark.parametrize(
    ("port", "adapter"),
    [(PaperSourcePort, ArxivPaperSource), (UserStoragePort, EnvUserStorage)],
    ids=["PaperSourcePort", "UserStoragePort"],
)
def test_adapters_without_a_fake_match_their_port(port, adapter):
    """These ports have no test double, so the adapter is checked on its own."""
    for method in _abstract_methods(port):
        assert _parameters(getattr(adapter, method)) == _parameters(getattr(port, method)), (
            f"{adapter.__name__}.{method} does not match {port.__name__}"
        )


class TestVectorStoreFilterContract:
    """The specific contract whose drift caused a production 500."""

    @pytest.mark.asyncio
    async def test_fake_rejects_a_shape_the_real_adapter_cannot_bind(self):
        """PostgresVectorStore binds paper_ids as $2::uuid[]; a dict cannot encode."""
        store = MockVectorStorePort(chunks=[])

        with pytest.raises(TypeError):
            await store.search([0.1] * 384, top_k=5, paper_ids={"$in": ["paper-001"]})

    @pytest.mark.asyncio
    async def test_scoping_is_applied_before_the_limit(self, mixed_paper_chunks):
        """Filtering after slicing would silently return fewer than top_k matches.

        The real adapter filters in SQL and then LIMITs, so the fake must too.
        """
        store = MockVectorStorePort(chunks=mixed_paper_chunks)

        results = await store.search([0.1] * 384, top_k=2, paper_ids=["paper-002"])

        assert len(results) == 2
        assert {chunk.paper_id for chunk, _ in results} == {"paper-002"}


class TestArchitectureRules:
    """The rules CLAUDE.md advertises, checked mechanically.

    A documented rule nobody verifies is a claim, not a constraint — the domain
    layer was described as importing "only stdlib" while every entity imported
    pydantic.
    """

    ALLOWED_DOMAIN_PACKAGES = {"pydantic", "src"}

    def _imported_roots(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text())
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        return roots

    def _python_files(self, *parts: str) -> list[Path]:
        root = Path(__file__).resolve().parent.parent / "src"
        return sorted(root.joinpath(*parts).rglob("*.py"))

    def test_domain_imports_no_sdk_driver_or_framework(self):
        offenders: dict[str, set[str]] = {}
        for path in self._python_files("domain"):
            external = {
                root
                for root in self._imported_roots(path)
                if root not in self.ALLOWED_DOMAIN_PACKAGES and root not in sys.stdlib_module_names
            }
            if external:
                offenders[str(path.name)] = external

        assert not offenders, f"domain layer imports external packages: {offenders}"

    def test_application_never_imports_adapters(self):
        offenders: dict[str, list[str]] = {}
        for path in self._python_files("application"):
            bad = [
                node.module
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src.adapters")
            ]
            if bad:
                offenders[path.name] = bad

        assert not offenders, f"application layer reaches into adapters: {offenders}"

    def test_sdk_imports_stay_in_outbound_adapters(self):
        sdk_packages = {"anthropic", "asyncpg", "arxiv", "fastembed", "fitz", "umap", "hdbscan"}
        offenders: dict[str, set[str]] = {}
        for path in self._python_files():
            if "adapters/outbound" in path.as_posix():
                continue
            used = self._imported_roots(path) & sdk_packages
            if used:
                offenders[path.as_posix().split("src/")[-1]] = used

        assert not offenders, f"SDK imports outside outbound adapters: {offenders}"

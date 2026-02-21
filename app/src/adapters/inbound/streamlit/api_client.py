"""HTTP client for communicating with the FastAPI backend."""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class HealthStatus:
    """Health check response."""

    status: str
    stats: dict[str, Any]


@dataclass
class PaperInfo:
    """Paper information from the backend."""

    paper_id: str
    arxiv_id: str
    title: str
    chunk_count: int


@dataclass
class IngestionResult:
    """Result of paper ingestion."""

    success: list[str]
    failed: list[dict[str, str]]


class APIClient:
    """Client for communicating with the ExplainRAG FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        """Initialize the API client.

        Args:
            base_url: Base URL of the FastAPI backend.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_client(self) -> httpx.Client:
        """Get a configured HTTP client."""
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health_check(self) -> HealthStatus:
        """Check the health of the backend.

        Returns:
            HealthStatus with status and stats.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        with self._get_client() as client:
            response = client.get("/health")
            response.raise_for_status()
            data = response.json()
            return HealthStatus(
                status=data.get("status", "unknown"),
                stats=data.get("stats", {}),
            )

    def list_papers(self) -> list[PaperInfo]:
        """List all ingested papers.

        Returns:
            List of PaperInfo objects.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        with self._get_client() as client:
            response = client.get("/papers")
            response.raise_for_status()
            data = response.json()
            return [
                PaperInfo(
                    paper_id=p.get("paper_id", ""),
                    arxiv_id=p.get("arxiv_id", ""),
                    title=p.get("title", "Unknown"),
                    chunk_count=p.get("chunk_count", 0),
                )
                for p in data.get("papers", [])
            ]

    def delete_paper(self, paper_id: str) -> dict[str, Any]:
        """Delete a paper and all its chunks.

        Args:
            paper_id: The paper ID to delete.

        Returns:
            Dictionary with paper_id and deleted_chunks count.

        Raises:
            httpx.HTTPError: If the request fails or paper not found (404).
        """
        with self._get_client() as client:
            response = client.delete(f"/papers/{paper_id}")
            response.raise_for_status()
            return response.json()

    def ingest_papers(self, arxiv_ids: list[str]) -> IngestionResult:
        """Ingest papers by arXiv IDs.

        Args:
            arxiv_ids: List of arXiv IDs to ingest.

        Returns:
            IngestionResult with success and failed lists.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        with self._get_client() as client:
            response = client.post(
                "/ingest",
                json={"arxiv_ids": arxiv_ids},
            )
            response.raise_for_status()
            data = response.json()
            return IngestionResult(
                success=data.get("success", []),
                failed=data.get("failed", []),
            )

    def query(
        self,
        question: str,
        top_k: int = 10,
        paper_ids: list[str] | None = None,
        enable_reranking: bool = False,
    ) -> dict[str, Any]:
        """Send a query to the backend.

        Args:
            question: The natural language question.
            top_k: Number of chunks to retrieve.
            paper_ids: Optional list of paper IDs to filter.
            enable_reranking: Whether to enable cross-encoder reranking.

        Returns:
            Full QueryResponse as a dictionary.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        payload: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "enable_reranking": enable_reranking,
        }
        if paper_ids:
            payload["paper_ids"] = paper_ids

        with self._get_client() as client:
            response = client.post("/query", json=payload)
            response.raise_for_status()
            return response.json()

    def get_explanation(self, query_id: str) -> dict[str, Any]:
        """Get the explanation for a previous query.

        Args:
            query_id: The query UUID.

        Returns:
            Full QueryResponse as a dictionary.

        Raises:
            httpx.HTTPError: If the request fails or query not found.
        """
        with self._get_client() as client:
            response = client.get(f"/query/{query_id}/explanation")
            response.raise_for_status()
            return response.json()

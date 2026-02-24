from abc import ABC, abstractmethod

from src.domain.entities.query import QueryResponse


class QueryNotFoundError(Exception):
    """Raised when a query is not found in storage."""

    pass


class QueryStoragePort(ABC):
    """Abstract interface for query persistence operations."""

    @abstractmethod
    async def store(self, response: QueryResponse) -> None:
        """Store a query response.

        Args:
            response: The QueryResponse to store.
        """
        ...

    @abstractmethod
    async def get(self, query_id: str) -> QueryResponse | None:
        """Retrieve a query response by ID.

        Args:
            query_id: The query UUID.

        Returns:
            The QueryResponse if found, None otherwise.
        """
        ...

    @abstractmethod
    async def list_recent(self, limit: int = 20) -> list[dict]:
        """List recent queries with summary information.

        Args:
            limit: Maximum number of queries to return.

        Returns:
            List of query summaries with id, question, answer preview, and created_at.
        """
        ...

    @abstractmethod
    async def delete(self, query_id: str) -> bool:
        """Delete a query from storage.

        Args:
            query_id: The query UUID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get the total number of stored queries.

        Returns:
            Total count of queries in storage.
        """
        ...

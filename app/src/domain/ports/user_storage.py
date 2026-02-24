"""User storage port for authentication."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class User:
    """User entity for authentication."""

    id: str
    username: str
    hashed_password: str
    is_admin: bool = True


class UserStoragePort(ABC):
    """Abstract interface for user storage operations."""

    @abstractmethod
    async def get_user_by_username(self, username: str) -> User | None:
        """Retrieve a user by username.

        Args:
            username: The username to look up.

        Returns:
            User if found, None otherwise.
        """
        ...

    @abstractmethod
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash.

        Args:
            plain_password: The plain text password.
            hashed_password: The bcrypt hash to verify against.

        Returns:
            True if password matches, False otherwise.
        """
        ...

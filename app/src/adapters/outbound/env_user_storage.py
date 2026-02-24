"""Environment-based user storage adapter."""

import asyncio
import logging

import bcrypt

from src.domain.ports.user_storage import User, UserStoragePort

logger = logging.getLogger(__name__)


class EnvUserStorage(UserStoragePort):
    """User storage that reads admin credentials from environment variables."""

    def __init__(self, admin_username: str, admin_password_hash: str):
        """Initialize the environment user storage.

        Args:
            admin_username: The admin username from settings.
            admin_password_hash: The bcrypt hash of admin password.
        """
        self._admin_username = admin_username
        self._admin_password_hash = admin_password_hash

    async def get_user_by_username(self, username: str) -> User | None:
        """Retrieve a user by username.

        Returns the configured admin user if username matches.
        """
        if username == self._admin_username and self._admin_password_hash:
            return User(
                id="admin",
                username=self._admin_username,
                hashed_password=self._admin_password_hash,
                is_admin=True,
            )
        return None

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a bcrypt hash."""
        return await asyncio.to_thread(
            bcrypt.checkpw,
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )

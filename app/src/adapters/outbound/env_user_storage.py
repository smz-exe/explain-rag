"""Environment-based user storage adapter."""

import asyncio
import logging
import secrets

import bcrypt

from src.domain.ports.user_storage import User, UserStoragePort

logger = logging.getLogger(__name__)

# Compared against when a username is unknown, so login spends the same bcrypt
# time on both branches. Derived from a fresh random value at import: it is not
# a credential and nothing can match it.
_DUMMY_PASSWORD_HASH = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt(12)).decode()


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

    async def authenticate(self, username: str, password: str) -> User | None:
        """Check a username and password in constant-ish time.

        The bcrypt comparison runs whether or not the username exists, so an
        unknown user cannot be distinguished from a wrong password by latency.
        """
        user = await self.get_user_by_username(username)
        hashed = user.hashed_password if user else _DUMMY_PASSWORD_HASH
        password_matches = await self.verify_password(password, hashed)
        return user if (user and password_matches) else None

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a bcrypt hash."""
        return await asyncio.to_thread(
            bcrypt.checkpw,
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )

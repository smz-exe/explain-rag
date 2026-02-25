from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    anthropic_api_key: SecretStr = SecretStr("")
    claude_model: str = "claude-sonnet-4-5-20250929"
    claude_max_tokens: int = 4096
    claude_timeout: float = 120.0  # Timeout in seconds for Claude API calls
    claude_max_retries: int = 2  # Max retries for transient failures

    # Embedding Configuration
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: SecretStr = SecretStr("")

    # Retrieval Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 10
    chroma_persist_dir: str = "./data/chroma"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Storage Configuration
    sqlite_db_path: str = "./data/queries.db"

    # Model Loading Configuration
    preload_models: bool = True  # Preload models at startup to avoid cold start
    hf_offline_mode: bool = False  # Use only locally cached HuggingFace models
    hf_token: SecretStr = SecretStr("")  # HuggingFace token for higher rate limits

    # Visualization Configuration (UMAP)
    umap_n_neighbors: int = 15  # Number of neighbors for UMAP
    umap_min_dist: float = 0.1  # Minimum distance between points in low-dimensional space

    # Clustering Configuration (HDBSCAN)
    hdbscan_min_cluster_size: int = 2  # Minimum cluster size
    hdbscan_min_samples: int = 1  # Minimum samples for core points

    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    environment: Literal["development", "production"] = "development"

    # Auth Configuration
    jwt_secret_key: SecretStr = SecretStr("")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    admin_username: str = "admin"
    admin_password_hash: SecretStr = SecretStr("")  # bcrypt hash

    @property
    def secure_cookies(self) -> bool:
        """Use secure cookies in production (requires HTTPS)."""
        return self.environment == "production"

    @model_validator(mode="after")
    def validate_required_secrets(self) -> "Settings":
        """Validate required secrets are configured properly."""
        # Validate JWT secret
        jwt_value = self.jwt_secret_key.get_secret_value()
        if not jwt_value or len(jwt_value) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be set and at least 32 characters. "
                'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        # Validate Anthropic API key
        if not self.anthropic_api_key.get_secret_value():
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Get your API key from https://console.anthropic.com/"
            )

        # Validate admin password hash
        if not self.admin_password_hash.get_secret_value():
            raise ValueError(
                "ADMIN_PASSWORD_HASH is required. "
                'Generate with: python -c "import bcrypt; print(bcrypt.hashpw(b\'password\', bcrypt.gensalt(12)).decode())"'
            )

        return self

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
    claude_model: str = "claude-sonnet-5"
    claude_max_tokens: int = 4096
    claude_timeout: float = 120.0  # Timeout in seconds for Claude API calls
    claude_max_retries: int = 2  # Max retries for transient failures

    # Defer faithfulness verification to a background task so /query returns
    # as soon as the answer is generated (verification dominates latency)
    deferred_verification: bool = True

    # HTTP hardening
    max_request_body_bytes: int = 1_048_576  # Reject request bodies larger than 1 MiB

    # Research Atlas: recompute UMAP coordinates on startup when the cache is
    # empty (production cache storage is ephemeral and wiped by deploys)
    recompute_coordinates_on_startup: bool = True

    # Embedding Configuration (FastEmbed with ONNX Runtime)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # FastEmbed format

    # Retrieval Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 10
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # FastEmbed format

    # Storage Configuration
    sqlite_db_path: str = "./data/queries.db"

    # Database Configuration (Supabase / PostgreSQL) — required
    # Format: postgresql://user:password@host:port/database
    database_url: str = ""
    database_pool_min: int = 2
    database_pool_max: int = 10
    # Bound every database interaction: without these a stalled connection or an
    # exhausted pool hangs the request until the client gives up.
    database_command_timeout: float = 30.0  # Max seconds for a single statement
    database_acquire_timeout: float = 10.0  # Max seconds waiting for a connection

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
    # Note: the listen address and port are set by the Dockerfile's uvicorn
    # command, not from here — a setting nothing reads would only mislead.
    cors_origins: list[str] = ["http://localhost:3000"]
    environment: Literal["development", "production"] = "development"

    # Rate Limiting Configuration
    rate_limit_query: str = "10/minute"  # Rate limit for /query endpoint
    rate_limit_login: str = "5/minute"  # Rate limit for /auth/login (brute force)
    rate_limit_enabled: bool = True  # Enable/disable rate limiting
    # Header carrying the real client IP when running behind a trusted proxy.
    # Left unset by default: reading a client-supplied header would let anyone
    # forge their rate-limit identity. Set it only in a deployment where the
    # proxy overwrites the header (on Fly: CLIENT_IP_HEADER=Fly-Client-IP).
    client_ip_header: str | None = None

    # Auth Configuration
    jwt_secret_key: SecretStr = SecretStr("")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    admin_username: str = "admin"
    admin_password_hash: SecretStr = SecretStr("")  # bcrypt hash

    # Capability token granting read access to a single stored query.
    # Short-lived: it only has to outlive the session that asked the question.
    query_token_expire_minutes: int = 120

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

        # Validate database URL (PostgreSQL/pgvector is the only vector store)
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is required (PostgreSQL with pgvector). "
                "For local development, run 'supabase start' and use the local URL."
            )

        # Validate admin password hash
        if not self.admin_password_hash.get_secret_value():
            raise ValueError(
                "ADMIN_PASSWORD_HASH is required. "
                "Generate with: python -c \"import bcrypt; print(bcrypt.hashpw(b'password', bcrypt.gensalt(12)).decode())\""
            )

        return self

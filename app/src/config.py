from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5-20250929"
    claude_max_tokens: int = 4096

    # Embedding Configuration
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str = ""

    # Retrieval Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 10
    chroma_persist_dir: str = "./data/chroma"

    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

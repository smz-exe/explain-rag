# ExplainRAG Backend

Explainable Retrieval-Augmented Generation for academic papers.

## Setup

```bash
cd app
uv sync
cp .env.example .env
# Edit .env with your API keys
```

## Running

```bash
# Start FastAPI server
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v
```

## API Documentation

Once running, visit http://localhost:8000/docs for the OpenAPI documentation.

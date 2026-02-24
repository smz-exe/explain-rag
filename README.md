# ExplainRAG

Explainable Retrieval-Augmented Generation for academic papers.

## Quick Start

### Backend (FastAPI)

```bash
cd app

# Install dependencies
uv sync --extra dev

# Run server
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v
```

### Frontend (Next.js)

```bash
cd frontend

# Install dependencies
pnpm install

# Generate API types from backend
pnpm generate:api

# Run dev server
pnpm dev

# Run E2E tests
pnpm test:e2e
```

## Usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest a paper
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["1706.03762"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?"}'
```

- **API docs:** <http://localhost:8000/docs>
- **Frontend:** <http://localhost:3000>

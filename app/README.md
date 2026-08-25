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

## Batch Paper Ingestion (CLI)

Ingest arXiv papers from the command line without going through the HTTP API.
The pipeline (arXiv fetch → PDF chunking → local embedding → pgvector insert)
runs on your machine; `--env-file` selects the target database, so pointing it
at `.env.production` writes to the production Supabase instance. Papers already
in the store are skipped automatically.

```bash
# Specific papers into the local database
uv run python -m src.adapters.inbound.cli.ingest 2005.11401 1810.04805

# Curated seed corpus into production
uv run python -m src.adapters.inbound.cli.ingest --file seeds/papers.txt --env-file .env.production

# Search arXiv and ingest the top matches
uv run python -m src.adapters.inbound.cli.ingest --search "retrieval augmented generation" --max-results 5

# Preview what would be ingested without writing anything
uv run python -m src.adapters.inbound.cli.ingest --file seeds/papers.txt --dry-run
```

After ingesting into production, refresh the Research Atlas from the admin page
(recompute coordinates) so the new papers appear in the visualization.

## API Documentation

Once running, visit http://localhost:8000/docs for the OpenAPI documentation.

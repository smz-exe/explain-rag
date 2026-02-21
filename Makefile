.PHONY: dev frontend backend

frontend:
	cd frontend && pnpm dev

backend:
	cd app && uv run uvicorn src.main:app --reload --port 8000

dev:
	@make -j2 backend frontend
# ── MedRAG Pipeline — Makefile ────────────────────────────────────────────────
# Usage: make <target>

.PHONY: help install test test-cov lint format run-ui run-api pipeline clean docker-up

PYTHON  = python
UVICORN = uvicorn
STREAMLIT = streamlit

# Default query for the pipeline
QUERY   ?= "type 2 diabetes treatment metformin"
DRUGS   ?= metformin semaglutide
MAX     ?= 30

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

test: ## Run test suite (no coverage)
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Lint with ruff
	ruff check src/ tests/ scripts/ app.py

format: ## Format with black
	black src/ tests/ scripts/ app.py

pipeline: ## Run the ETL pipeline (set QUERY, DRUGS, MAX to override)
	$(PYTHON) scripts/run_pipeline.py \
		--query $(QUERY) \
		--max_results $(MAX) \
		--drugs $(DRUGS)

run-ui: ## Launch the Streamlit UI
	$(STREAMLIT) run app.py

run-api: ## Start the FastAPI server (development mode)
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

query: ## Quick query (set Q= to override)
	$(PYTHON) scripts/query_agent.py $(Q)

interactive: ## Start interactive CLI session
	$(PYTHON) scripts/query_agent.py --interactive

docker-up: ## Start full stack with Docker Compose
	docker-compose up --build

docker-down: ## Stop Docker Compose services
	docker-compose down

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	@echo "Cleaned."

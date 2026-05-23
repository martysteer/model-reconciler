SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := up

.PHONY: up stop test lint logs clean

up:
	@echo "Model Reconciler — http://127.0.0.1:8001/"
	@docker compose up --build

stop:
	@docker compose down

test:
	@docker compose run --rm --no-deps \
		-v "./tests:/app/tests:ro" \
		api pytest tests/ -v

lint:
	@docker compose run --rm --no-deps \
		-v "./tests:/app/tests:ro" \
		api ruff check src/ tests/

logs:
	@docker compose logs -f

clean:
	@docker compose down -v --remove-orphans 2>/dev/null || true
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov

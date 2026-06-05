# ============================================================================
# Docker Commands
# ============================================================================

.PHONY: help build build-gpu up down test jupyter clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build commands
build: ## Build CPU Docker image
	docker build -t author-switch-detector:latest .

build-gpu: ## Build GPU Docker image
	docker build -f Dockerfile.gpu -t author-switch-detector:gpu .

# Run commands
up: ## Start the application
	docker-compose up app

up-gpu: ## Start the GPU application
	docker-compose -f docker-compose.gpu.yml up app-gpu

jupyter: ## Start Jupyter notebook
	docker-compose up jupyter

jupyter-gpu: ## Start GPU Jupyter notebook
	docker-compose -f docker-compose.gpu.yml up jupyter-gpu

# Development
dev: ## Start development environment (app + jupyter)
	docker-compose up app jupyter

test: ## Run tests
	docker-compose run --rm test

# Cleanup
down: ## Stop all services
	docker-compose down

clean: ## Remove Docker images and containers
	docker-compose down -v --rmi all
	docker system prune -f

# Quick run without Docker
run-local: ## Run locally (requires virtual environment)
	PYTHONPATH=./src python -m src.main

train-local: ## Train models locally
	PYTHONPATH=./src python -m src.main --mode train

evaluate-local: ## Evaluate models locally
	PYTHONPATH=./src python -m src.main --mode evaluate

# Docker interactive
shell: ## Open shell in container
	docker-compose run --rm app bash

shell-gpu: ## Open shell in GPU container
	docker-compose -f docker-compose.gpu.yml run --rm app-gpu bash

# Data management
download-nltk: ## Download NLTK data manually
	python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
	
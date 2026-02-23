.PHONY: help install install-dev test lint format clean train evaluate docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install the package with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  train        Run training"
	@echo "  evaluate     Run evaluation"
	@echo "  docs         Build documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=pytorch_transformers --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 pytorch_transformers/ tests/
	mypy pytorch_transformers/

format:
	black pytorch_transformers/ tests/ scripts/
	isort pytorch_transformers/ tests/ scripts/

format-check:
	black --check pytorch_transformers/ tests/ scripts/
	isort --check-only pytorch_transformers/ tests/ scripts/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Training and evaluation
train:
	python -m pytorch_transformers.scripts.train

evaluate:
	python -m pytorch_transformers.scripts.evaluate

# Documentation
docs:
	@echo "Documentation building not yet implemented"

# Development helpers
dev-setup: install-dev
	@echo "Development environment setup complete!"

check: lint test
	@echo "All checks passed!"

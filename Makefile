.PHONY: help install run-workbench run-server test clean build

help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  run-workbench - Run FOL Workbench GUI"
	@echo "  run-server    - Run MCP Server"
	@echo "  test          - Run tests"
	@echo "  build         - Build package"
	@echo "  clean         - Clean build artifacts"

install:
	@echo "Installing dependencies..."
	venv/bin/pip install -r requirements.txt

run-workbench:
	@echo "Running FOL Workbench..."
	PYTHONPATH=. venv/bin/python -m src.fol_workbench.main

run-server:
	@echo "Running MCP Server..."
	PYTHONPATH=. venv/bin/python -m mcp.server.fastmcp src.kaggle_mcp_server.server

test:
	@echo "Running tests..."
	PYTHONPATH=. venv/bin/python -m pytest

build:
	@echo "Building package..."
	venv/bin/pip install -e .

clean:
	@echo "Cleaning..."
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -r {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

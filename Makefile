.PHONY: help install run-workbench run-server run-moe-server generate-skills test clean build

help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies"
	@echo "  run-workbench    - Run FOL Workbench GUI"
	@echo "  run-server       - Run Kaggle MCP Server"
	@echo "  run-moe-server   - Run MLSysEng MoE MCP Server"
	@echo "  generate-skills  - Generate platform-specific skill configs"
	@echo "  test             - Run tests"
	@echo "  build            - Build package"
	@echo "  clean            - Clean build artifacts"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run-workbench:
	@echo "Running FOL Workbench..."
	PYTHONPATH=. python -m src.fol_workbench.main

run-server:
	@echo "Running Kaggle MCP Server..."
	PYTHONPATH=. python -m mcp.server.fastmcp src.kaggle_mcp_server.server

run-moe-server:
	@echo "Running MLSysEng MoE MCP Server..."
	PYTHONPATH=. python -m src.mlsyseng_mcp.server

generate-skills:
	@echo "Generating platform-specific skills..."
	python skill_generator.py

test:
	@echo "Running tests..."
	PYTHONPATH=. python -m pytest

build:
	@echo "Building package..."
	pip install -e .

clean:
	@echo "Cleaning..."
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -r {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

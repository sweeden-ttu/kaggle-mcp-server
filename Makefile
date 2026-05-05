.PHONY: help install run-workbench run-server run-mlsyseng test clean build generate-skills

help:
	@echo "Available targets:"
	@echo "  install         - Install dependencies"
	@echo "  run-workbench   - Run FOL Workbench GUI"
	@echo "  run-server      - Run Kaggle MCP Server"
	@echo "  run-mlsyseng    - Run MLSysEng MoE MCP Server"
	@echo "  generate-skills - Generate platform-specific skill configs"
	@echo "  test            - Run tests"
	@echo "  build           - Build package"
	@echo "  clean           - Clean build artifacts"

install:
	@echo "Installing dependencies..."
	venv/bin/pip install -r requirements.txt

run-workbench:
	@echo "Running FOL Workbench..."
	PYTHONPATH=. venv/bin/python -m src.fol_workbench.main

run-server:
	@echo "Running Kaggle MCP Server..."
	PYTHONPATH=. venv/bin/python -m mcp.server.fastmcp src.kaggle_mcp_server.server

run-mlsyseng:
	@echo "Running MLSysEng MoE MCP Server..."
	PYTHONPATH=. venv/bin/python -m src.mlsyseng_mcp.server

generate-skills:
	@echo "Generating platform-specific skills..."
	PYTHONPATH=. venv/bin/python skill_generator.py

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

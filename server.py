"""Top-level server entry point for MLSysEng MoE MCP server.

Usage:
    python server.py
    python -m mlsyseng_mcp.server
"""

from mlsyseng_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()

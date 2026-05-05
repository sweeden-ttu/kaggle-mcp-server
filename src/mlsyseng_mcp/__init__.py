"""MLSysEng MoE - Mixture of Experts system for ML knowledge extraction and Kaggle competition building."""


def main():
    """Run the MLSysEng MoE MCP server."""
    from .server import main as _main

    _main()


__all__ = ["main"]

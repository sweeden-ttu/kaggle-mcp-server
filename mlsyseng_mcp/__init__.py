"""MLSysEng MoE - Machine Learning Systems Expert Mixture of Experts."""

__version__ = "0.1.0"


def main():
    """Run the MLSysEng MoE MCP server."""
    from .server import main as _main
    _main()


__all__ = ["main"]

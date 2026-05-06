"""MLSysEng MoE - Mixture of Experts system for ML Principles knowledge extraction."""

__version__ = "0.1.0"


def main():
    from .server import main as _main
    _main()


__all__ = ["main"]

"""MLSysEng MoE - Mixture of Experts system for ML Principles knowledge extraction."""


def main():
    from .server import main as _main

    _main()


__all__ = ["main"]

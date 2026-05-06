"""Package entry point for running as python -m mlsyseng_mcp.server."""

import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import database  # noqa: E402, F401
import docling_worker  # noqa: E402, F401
import embeddings  # noqa: E402, F401
import expert_registry  # noqa: E402, F401
import loop_controller  # noqa: E402, F401
from server import mcp, main  # noqa: E402, F401

if __name__ == "__main__":
    main()

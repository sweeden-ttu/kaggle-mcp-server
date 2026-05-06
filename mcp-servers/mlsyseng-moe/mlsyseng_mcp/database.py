"""Package re-export for database module."""

import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from database import *  # noqa: E402, F401, F403

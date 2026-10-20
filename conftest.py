"""pytest bootstrap: make the ``navfollow`` package importable from the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""Allow running tests/scripts without an editable install (``pip install -e .``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

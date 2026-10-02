"""Notebook entry point; keep environment details out of teaching cells."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jkj1a.runtime import prepare

_context = prepare(project_root=PROJECT_ROOT)
ON_COLAB = _context["ON_COLAB"]
project_root = _context["project_root"]

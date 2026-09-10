"""Shared pytest setup for the DARTree test suite.

Makes the repository root importable so tests can ``import utils.retrieval``
and ``import eval_dartree`` regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

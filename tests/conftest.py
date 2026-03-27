"""
Shared pytest configuration and fixtures for the YOLO-MSLesSeg test suite.

The pipeline uses relative paths anchored at the repository root
(e.g. Path("MSLesSeg-Dataset")), so all tests must run from that directory.
The session-scoped fixture below ensures this automatically.
"""

import os
from pathlib import Path

import pytest

# Repository root is two levels up from this file (tests/conftest.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def set_repo_root_as_cwd():
    """Change the working directory to the repository root for the entire test session."""
    original = Path.cwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(original)

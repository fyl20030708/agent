"""Tests for research_agent tools (no API key required)."""

from pathlib import Path

import pytest

from research_agent.tools import REPO_ROOT, list_experiment_dirs, list_python_files


@pytest.fixture
def repo_exists() -> bool:
    return REPO_ROOT.is_dir()


@pytest.mark.skipif(not Path("/homes/yli/31P_transformer").is_dir(), reason="31P repo not present")
def test_list_python_files_finds_training():
    out = list_python_files("Training", limit=10)
    assert "train.py" in out or ".py" in out


@pytest.mark.skipif(not Path("/homes/yli/31P_transformer").is_dir(), reason="31P repo not present")
def test_list_experiment_dirs():
    out = list_experiment_dirs(limit=5)
    assert "2026_" in out or out.startswith("(")

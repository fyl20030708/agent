import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week02"))

from react_loop import parse_action  # noqa: E402


def test_parse_list_dir() -> None:
    text = 'Thought: look\nAction: list_dir("Training")\n'
    assert parse_action(text) == ("list_dir", {"path": "Training"})


def test_parse_read_file() -> None:
    text = 'Thought: read\nAction: read_file("Training/train.py")\n'
    assert parse_action(text) == ("read_file", {"path": "Training/train.py"})


def test_parse_finish() -> None:
    text = 'Thought: done\nAction: finish("Training has train.py")\n'
    assert parse_action(text) == ("finish", {"answer": "Training has train.py"})


def test_parse_invalid_returns_none() -> None:
    assert parse_action("Thought: I am confused\n") is None

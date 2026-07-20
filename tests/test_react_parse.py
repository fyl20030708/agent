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


def test_parse_dirty_double_action_no_absurd_path() -> None:
    """Two Action lines before Observation must not swallow a huge path."""
    dirty = (
        "Thought: a\n"
        'Action: list_dir("Training")\n'
        "Thought: b\n"
        'Action: read_file("foo.py")\n'
        "Observation: ok\n"
    )
    result = parse_action(dirty)
    if result is not None:
        _, args = result
        path = args.get("path", args.get("answer", ""))
        assert len(path) < 200

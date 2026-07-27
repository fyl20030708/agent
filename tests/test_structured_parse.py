import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week03"))

from structured_output_demo import extract_json_object, validate_report  # noqa: E402


def test_extract_plain_json() -> None:
    text = '{"path": "Training/train.py", "summary": "trainer", "confidence": 0.9}'
    assert extract_json_object(text) == {
        "path": "Training/train.py",
        "summary": "trainer",
        "confidence": 0.9,
    }


def test_extract_json_from_fenced_block() -> None:
    fence = chr(96) * 3  # backtick * 3
    text = (
        "Here you go:\n"
        + fence
        + "json\n"
        + '{"path": "a.py", "summary": "x", "confidence": 0.5}\n'
        + fence
        + "\n"
    )
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["path"] == "a.py"


def test_extract_invalid_returns_none() -> None:
    assert extract_json_object("no json here") is None


def test_validate_report_ok() -> None:
    ok, msg = validate_report(
        {"path": "Training/train.py", "summary": "entry", "confidence": 0.8}
    )
    assert ok is True
    assert msg == ""


def test_validate_report_missing_key() -> None:
    ok, msg = validate_report({"path": "a.py", "summary": "x"})
    assert ok is False
    assert "confidence" in msg.lower() or "missing" in msg.lower()


def test_validate_report_bad_confidence() -> None:
    ok, msg = validate_report(
        {"path": "a.py", "summary": "x", "confidence": 2.0}
    )
    assert ok is False

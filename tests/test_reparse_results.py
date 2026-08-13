import json

from scripts.reparse_results import reparse_in_place


def test_reparse_updates_fields_and_preserves_raw_output_with_backup(tmp_path):
    path = tmp_path / "raw.jsonl"
    backup = tmp_path / "raw.backup.jsonl"
    raw = "Reasoning Score: 20.assistantfinalScore: 95"
    original = {
        "raw_output": raw,
        "score": None,
        "score_parse_ok": False,
        "score_parse_error": "no_labelled_score",
        "format_ok": False,
        "format_error": "expected_exact_score_format",
    }
    path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    summary = reparse_in_place(path, backup)
    updated = json.loads(path.read_text(encoding="utf-8"))
    preserved = json.loads(backup.read_text(encoding="utf-8"))

    assert summary == {
        "rows": 1,
        "changed_rows": 1,
        "score_parse_ok": 1,
        "format_ok": 1,
    }
    assert updated["raw_output"] == raw
    assert updated["score"] == 95
    assert updated["score_parse_ok"] is True
    assert updated["format_ok"] is True
    assert preserved == original

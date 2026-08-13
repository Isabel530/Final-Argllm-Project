#!/usr/bin/env python3
"""Recompute parser fields in a raw JSONL without regenerating model output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.parsing import parse_score  # noqa: E402


PARSER_FIELDS = (
    "score",
    "score_parse_ok",
    "score_parse_error",
    "format_ok",
    "format_error",
)


def _reparsed_fields(raw_output: str) -> dict[str, Any]:
    strict = parse_score(raw_output, strict=True)
    soft = parse_score(raw_output, strict=False)
    return {
        "score": soft.score,
        "score_parse_ok": soft.ok,
        "score_parse_error": soft.error,
        "format_ok": strict.ok,
        "format_error": strict.error,
    }


def reparse_in_place(path: Path, backup: Path) -> dict[str, int]:
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if "raw_output" not in row:
                raise ValueError(f"{path}:{line_number}: missing raw_output")
            rows.append(row)

    original_mode = stat.S_IMODE(path.stat().st_mode)
    shutil.copy2(path, backup)
    changed_rows = 0
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for row in rows:
                updated = _reparsed_fields(row["raw_output"])
                if any(row.get(field) != updated[field] for field in PARSER_FIELDS):
                    changed_rows += 1
                row.update(updated)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, original_mode)
        os.replace(temp_name, path)
    except BaseException:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise

    return {
        "rows": len(rows),
        "changed_rows": changed_rows,
        "score_parse_ok": sum(bool(row["score_parse_ok"]) for row in rows),
        "format_ok": sum(bool(row["format_ok"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()

    backup = args.backup or args.input.with_name(
        args.input.name + ".pre-final-channel-parser.bak"
    )
    summary = reparse_in_place(args.input, backup)
    print(json.dumps({"input": str(args.input), "backup": str(backup), **summary}))


if __name__ == "__main__":
    main()

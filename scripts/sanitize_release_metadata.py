#!/usr/bin/env python
"""Audit and optionally replace machine-specific paths in release metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml", ".example"}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}

PATTERNS = [
    (re.compile(r"[A-Za-z]:\\(?:[^\\;|\r\n]+\\)*Saluki(?:\\paper\\mrna-half-life-complementarity)?", re.I), "."),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\\r\n]+", re.I), "<user-home>"),
    (re.compile(r"/s/R1/[^\"'\r\n,\s]+"), "."),
    (re.compile(r"/data1/[^\"'\r\n,\s]+"), "./results/runs"),
]


def candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.endswith(".env.example"):
            yield path


def apply_patterns(text: str) -> tuple[str, int]:
    updated = text
    count = 0
    for pattern, replacement in PATTERNS:
        updated, n = pattern.subn(replacement, updated)
        count += n
    return updated, count


def sanitize_workbooks(write: bool) -> list[tuple[Path, int]]:
    from openpyxl import load_workbook

    findings = []
    for path in (ROOT / "tables").glob("*.xlsx"):
        workbook = load_workbook(path)
        count = 0
        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if not isinstance(cell.value, str):
                        continue
                    updated, n = apply_patterns(cell.value)
                    count += n
                    if write and n:
                        cell.value = updated
        if count:
            findings.append((path.relative_to(ROOT), count))
            if write:
                workbook.save(path)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write replacements; default is audit-only.")
    args = parser.parse_args()

    findings = []
    for path in candidate_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        updated, count = apply_patterns(text)
        if count:
            findings.append((path.relative_to(ROOT), count))
            if args.write:
                path.write_text(updated, encoding="utf-8")

    findings.extend(sanitize_workbooks(args.write))

    action = "replaced" if args.write else "found"
    print(f"Machine-specific path strings {action}: {sum(n for _, n in findings)}")
    for path, count in findings:
        print(f"  {path}: {count}")
    return 0 if args.write or not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())

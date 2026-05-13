#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate (and optionally insert) SPDX headers.

Replacement for espressif/check-copyright that supports the full SPDX
expression grammar (AND, OR, WITH, parentheses) via the license-expression
library. Reads the same .github/check-spdx.yaml shape and is wired in as a
local pre-commit hook.

When a file is missing both SPDX-FileCopyrightText and SPDX-License-Identifier
headers, the configured template (`new_notice_python` or `new_notice_c`) is
inserted at the top of the file (after any shebang). Pre-commit then reports
the file as modified and the user re-stages + re-runs.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import yaml
from license_expression import ExpressionError, get_spdx_licensing

LICENSE_RE = re.compile(
    r"^[^\w\n]*SPDX-License-Identifier:\s*(.+?)\s*$", re.MULTILINE
)
COPYRIGHT_RE = re.compile(r"^[^\w\n]*SPDX-FileCopyrightText:", re.MULTILINE)

PYTHON_EXTS = {".py", ".pyi", ".pyx"}
C_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ld"}


def template_for(path: Path, section: dict) -> str | None:
    ext = path.suffix.lower()
    if ext in PYTHON_EXTS:
        return section.get("new_notice_python")
    if ext in C_EXTS:
        return section.get("new_notice_c")
    return None


def insert_header(path: Path, template: str, license_: str) -> bool:
    """Insert a formatted header at the top of the file (after any shebang).

    Returns True if the file was modified.
    """
    text = path.read_text(encoding="utf-8")
    header = template.format(
        license=license_, years=str(datetime.datetime.now().year)
    )
    if not header.endswith("\n"):
        header += "\n"

    shebang = ""
    body = text
    if text.startswith("#!"):
        nl = text.find("\n")
        if nl == -1:
            shebang, body = text + "\n", ""
        else:
            shebang, body = text[: nl + 1], text[nl + 1 :]

    new_text = shebang + header + body
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def check_file(
    path: Path, allowed: set[str], licensing, section: dict
) -> tuple[list[str], list[str]]:
    """Return (errors, modifications)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return [], []
    if not text.strip():
        return [], []

    has_copyright = bool(COPYRIGHT_RE.search(text))
    license_matches = LICENSE_RE.findall(text)

    if not has_copyright and not license_matches:
        template = template_for(path, section)
        license_ = section.get("license_for_new_files")
        if template and license_:
            if insert_header(path, template, license_):
                return [], [
                    f"{path}: inserted SPDX header — please review, "
                    f"stage, and re-run pre-commit."
                ]
        # Unknown file type or no insertion configured — fall through to error.

    errors: list[str] = []
    if not has_copyright:
        errors.append(f"{path}: missing SPDX-FileCopyrightText header")
    if not license_matches:
        errors.append(f"{path}: missing SPDX-License-Identifier header")
        return errors, []

    for expr in license_matches:
        try:
            parsed = licensing.parse(expr, validate=True, strict=True)
        except ExpressionError as e:
            errors.append(f"{path}: invalid SPDX expression {expr!r}: {e}")
            continue
        if parsed is None:
            errors.append(f"{path}: empty SPDX expression")
            continue
        atoms = [str(s) for s in parsed.symbols]
        bad = sorted({a for a in atoms if a not in allowed})
        if bad:
            errors.append(
                f"{path}: license(s) {bad} not in allowed list "
                f"{sorted(allowed)} (expression: {expr!r})"
            )

    return errors, []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("files", nargs="*", type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    section = cfg.get("DEFAULT", {}) or {}
    if not section.get("perform_check", True):
        return 0
    allowed = set(section.get("allowed_licenses", []))
    if not allowed:
        print(
            f"{args.config}: 'allowed_licenses' is empty — nothing would pass.",
            file=sys.stderr,
        )
        return 2

    licensing = get_spdx_licensing()
    errors: list[str] = []
    mods: list[str] = []
    for f in args.files:
        if f.is_file():
            e, m = check_file(f, allowed, licensing, section)
            errors.extend(e)
            mods.extend(m)

    for m in mods:
        print(m, file=sys.stderr)
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if (errors or mods) else 0


if __name__ == "__main__":
    sys.exit(main())

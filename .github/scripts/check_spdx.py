#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate SPDX-License-Identifier headers against an allowlist.

Replacement for espressif/check-copyright that supports the full SPDX
expression grammar (AND, OR, WITH, parentheses) via the license-expression
library. Reads the same .github/check-spdx.yaml shape and is wired in as a
local pre-commit hook.

Behavior vs. espressif/check-copyright:
  - Same: requires SPDX-FileCopyrightText and SPDX-License-Identifier headers;
    skips empty files.
  - Different: validation-only — does not auto-insert headers into files that
    lack them. Add headers manually.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from license_expression import ExpressionError, get_spdx_licensing

LICENSE_RE = re.compile(
    r"^[^\w\n]*SPDX-License-Identifier:\s*(.+?)\s*$", re.MULTILINE
)
COPYRIGHT_RE = re.compile(r"^[^\w\n]*SPDX-FileCopyrightText:", re.MULTILINE)


def check_file(path: Path, allowed: set[str], licensing) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if not text.strip():
        return []

    errors: list[str] = []
    if not COPYRIGHT_RE.search(text):
        errors.append(f"{path}: missing SPDX-FileCopyrightText header")

    license_matches = LICENSE_RE.findall(text)
    if not license_matches:
        errors.append(f"{path}: missing SPDX-License-Identifier header")
        return errors

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

    return errors


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
    for f in args.files:
        if f.is_file():
            errors.extend(check_file(f, allowed, licensing))

    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

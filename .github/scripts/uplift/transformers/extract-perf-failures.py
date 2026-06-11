# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract failure context from raw GHA perf job logs.

The perf pipeline (call-perf-test.yml) runs `pytest -svv` straight to
stdout with no junit XML / no log artifact. The workflow's
extract-failures step downloads the failed job logs via the actions API
and saves each one as `<model>__<runner>__job-<jid>.log` — the model
and runner are sourced from the GHA job name (the log BODY doesn't
contain that string, it's metadata on the job itself).

Outputs (mirror extract-failures.py's interface so the claude-fix step
is identical):
  --out-context        Human-readable failure summary fed to Claude.
  --out-failed-tests   One pytest command per line — pulled from the
                       perf-bench matrix by model name, so the next
                       iteration / a reviewer can re-run only the
                       failures with `pytest <commands>`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LOG_ERROR_PATTERNS = re.compile(
    r"(FAILED|ERROR|ModuleNotFoundError|ImportError|AttributeError"
    r"|NameError|TypeError.*argument|cannot import name|Traceback"
    r"|Samples/sec|expected fusion|regression detected)",
    re.IGNORECASE,
)

# Workflow saves files as "<model>__<runner>__job-<jid>.log".
FILENAME_PATTERN = re.compile(r"^(?P<model>.+?)__(?P<runner>[^_]+)__job-\d+$")

MAX_CHUNKS_PER_LOG = 30


def build_pytest_map(matrix_file: Path) -> dict[str, str]:
    """name -> pytest command, from .github/workflows/perf-bench-matrix.json."""
    pytest_map: dict[str, str] = {}
    if not matrix_file.exists():
        return pytest_map
    data = json.loads(matrix_file.read_text())
    for proj in data:
        for test in proj.get("tests", []):
            name = test.get("name")
            pytest_cmd = test.get("pytest")
            if name and pytest_cmd:
                pytest_map[name] = pytest_cmd
    return pytest_map


def extract_from_log(log_path: Path) -> tuple[str, str, list[str]]:
    """Returns (model, runner, error_chunks).

    model + runner are sourced from the filename. error_chunks are
    -2/+12 line windows around lines matching LOG_ERROR_PATTERNS,
    deduplicated by first 200 chars.
    """
    m = FILENAME_PATTERN.match(log_path.stem)
    if m:
        model, runner = m.group("model"), m.group("runner")
    else:
        model, runner = log_path.stem, "unknown"

    try:
        lines = log_path.read_text(errors="replace").splitlines(keepends=True)
    except OSError:
        return model, runner, []

    raw_chunks: list[str] = []
    for i, line in enumerate(lines):
        if LOG_ERROR_PATTERNS.search(line):
            start = max(0, i - 2)
            end = min(len(lines), i + 13)
            raw_chunks.append("".join(lines[start:end]))

    seen, unique = set(), []
    for c in raw_chunks:
        k = c[:200]
        if k in seen:
            continue
        seen.add(k)
        unique.append(c)
    return model, runner, unique[:MAX_CHUNKS_PER_LOG]


def format_output(
    results: list[tuple[str, str, list[str]]], pytest_map: dict[str, str]
) -> str:
    parts: list[str] = [f"=== FAILED PERF JOBS ({len(results)} total) ===\n"]
    for model, runner, chunks in results:
        parts.append(f"PERF: {model} ({runner})")
        cmd = pytest_map.get(model)
        if cmd:
            parts.append(f"PYTEST: {cmd}")
        else:
            parts.append("PYTEST: <unknown — model not in perf-bench-matrix.json>")
        if chunks:
            parts.append("ERROR EXCERPTS:")
            for c in chunks:
                parts.append(c)
                parts.append("~~~")
        parts.append("---")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--matrix-file", required=True)
    ap.add_argument("--out-context", required=True)
    ap.add_argument("--out-failed-tests", required=True)
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        print(f"::error::inputs dir not found: {inputs_dir}", file=sys.stderr)
        return 2

    pytest_map = build_pytest_map(Path(args.matrix_file))

    log_files = sorted(inputs_dir.rglob("*.log"))
    if not log_files:
        print(f"::warning::no log files under {inputs_dir}", file=sys.stderr)
        Path(args.out_context).write_text("")
        Path(args.out_failed_tests).write_text("")
        return 0

    # Dedup by model — same model failing on n150 + p150 yields two logs
    # but one rerun command suffices.
    results: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for log_file in log_files:
        model, runner, chunks = extract_from_log(log_file)
        if model in seen:
            continue
        seen.add(model)
        results.append((model, runner, chunks))

    Path(args.out_context).write_text(format_output(results, pytest_map))
    # last_failed.txt: one pytest command per line (rerunnable as-is).
    # Unknown models (no matrix entry) are skipped so the file stays
    # executable; they're still flagged in failure_context.txt.
    rerun_lines = [pytest_map[m] for m, _, _ in results if m in pytest_map]
    Path(args.out_failed_tests).write_text(
        "\n".join(rerun_lines) + ("\n" if rerun_lines else "")
    )

    print(
        f"Parsed {len(log_files)} log files; "
        f"{len(results)} unique failing perf jobs; "
        f"{len(rerun_lines)} pytest commands emitted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

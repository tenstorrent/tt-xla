#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Split a test config YAML into two files: one with passing tests and one with failing tests,
based on the top-level 'status' field of each test. Does not modify the original file.

Usage (from repo root, with venv activated):
  source venv/bin/activate
  python scripts/split_test_config_by_status.py [path/to/test_config_*.yaml]
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML

# Top-level status value that counts as "passing"; all others go to failing.
PASSING_STATUS = "EXPECTED_PASSING"


def _normalize_status(value: str) -> str:
    """Normalize status for comparison (enum name like EXPECTED_PASSING)."""
    if not value:
        return ""
    s = str(value).strip()
    # Accept enum name (EXPECTED_PASSING) or value (expected_passing)
    return s.upper() if s.islower() or "_" in s else s


def is_passing(status_value: str) -> bool:
    """Return True if this status is considered passing."""
    normalized = _normalize_status(status_value)
    return normalized == PASSING_STATUS or normalized == "EXPECTED_PASSING"


def get_top_level_status(entry: dict) -> str | None:
    """Get the top-level 'status' of a test config entry."""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("status")
    if raw is None:
        return None
    return str(raw).strip()


def _read_header(input_path: Path) -> str:
    """Return all lines before the 'test_config:' line (comment block / copyright)."""
    header_lines = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() == "test_config:":
                break
            header_lines.append(line)
    return "".join(header_lines)


def split_config_by_status(
    input_path: Path,
    passing_path: Path,
    failing_path: Path,
) -> tuple[int, int]:
    """
    Load input YAML, split test_config by status, write passing and failing YAMLs.
    Returns (num_passing, num_failing).
    """
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096

    header = _read_header(input_path)

    with open(input_path, "r") as f:
        data = yaml.load(f)

    if not data or "test_config" not in data:
        raise SystemExit(f"No 'test_config' key in {input_path}")

    test_config = data["test_config"]
    if not hasattr(test_config, "items"):
        raise SystemExit("'test_config' is not a mapping")

    passing = {}
    failing = {}

    for test_id, entry in test_config.items():
        status = get_top_level_status(entry)
        if status is None:
            # No top-level status: treat as failing to be safe
            failing[test_id] = entry
            continue
        if is_passing(status):
            passing[test_id] = entry
        else:
            failing[test_id] = entry

    passing_doc = {"test_config": passing}
    failing_doc = {"test_config": failing}

    buf = StringIO()
    yaml.dump(passing_doc, buf)
    with open(passing_path, "w") as f:
        f.write(header)
        f.write(buf.getvalue())

    buf = StringIO()
    yaml.dump(failing_doc, buf)
    with open(failing_path, "w") as f:
        f.write(header)
        f.write(buf.getvalue())

    return len(passing), len(failing)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split test config YAML into passing and failing files by status."
    )
    parser.add_argument(
        "input_yaml",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent
        / "tests/runner/test_config/torch/test_config_inference_single_device.yaml",
        help="Input test config YAML (default: tests/runner/test_config/torch/test_config_inference_single_device.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same as input)",
    )
    args = parser.parse_args()

    input_path = args.input_yaml.resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    passing_path = out_dir / f"{stem}_passing.yaml"
    failing_path = out_dir / f"{stem}_failing.yaml"

    try:
        num_passing, num_failing = split_config_by_status(
            input_path, passing_path, failing_path
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Read {num_passing + num_failing} tests from {input_path}")
    print(f"  Passing: {num_passing} -> {passing_path}")
    print(f"  Failing: {num_failing} -> {failing_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

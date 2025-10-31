#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verify that every test listed in missing_tests.log appears in the data-parallel config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_PATH = PROJECT_ROOT / "missing_tests.log"
DEFAULT_YAML_PATH = (
    PROJECT_ROOT / "tests/runner/test_config/test_config_inference_data_parallel.yaml"
)


def load_tests_from_log(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing tests log not found: {path}")

    tests: list[str] = []
    capture_next = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        if not line:
            capture_next = False
            continue

        if line.startswith("#"):
            continue

        if line.startswith("Missing data-parallel entry:") or line.startswith(
            "Bringup mismatch in data-parallel:"
        ):
            capture_next = True
            continue

        if capture_next:
            tests.append(line)
            capture_next = False

    return tests


def load_data_parallel_keys(yaml_path: Path) -> set[str]:
    yaml = YAML()
    data = yaml.load(yaml_path.read_text())
    if not isinstance(data, CommentedMap):
        raise ValueError(f"Unexpected YAML structure in {yaml_path}")

    test_config = data.get("test_config")
    if not isinstance(test_config, CommentedMap):
        raise ValueError("Missing 'test_config' mapping in data-parallel YAML")

    return set(test_config.keys())


def insert_tests(yaml_path: Path, tests: list[str]) -> None:
    tests = list(dict.fromkeys(tests))
    if not tests:
        return

    text = yaml_path.read_text()
    pattern = re.compile(
        r"(#==============================================================================\n\s*# TO BE TESTED\n#==============================================================================\n)"
    )

    match = pattern.search(text)
    if not match:
        raise ValueError("Could not locate 'TO BE TESTED' section in YAML file")

    block_lines: list[str] = []
    for name in tests:
        block_lines.append(f"  {name}:")
        block_lines.append("    supported_archs: [n300]")
        block_lines.append("    status: EXPECTED_PASSING")
        block_lines.append("")

    block = "\n".join(block_lines)

    new_text = text[: match.end()] + block + text[match.end() :]
    yaml_path.write_text(new_text)


def check_presence(log_path: Path, yaml_path: Path) -> tuple[int, list[str]]:
    tests = load_tests_from_log(log_path)
    existing_keys = load_data_parallel_keys(yaml_path)

    missing: list[str] = []
    skipped: list[str] = []
    converted_total = 0

    for test in tests:
        if "-single_device-" not in test:
            skipped.append(test)
            continue

        converted_total += 1
        dp_name = test.replace("-single_device-", "-data_parallel-", 1)

        if dp_name not in existing_keys:
            missing.append(dp_name)

    if missing:
        print("Tests not found in data-parallel config:")
        for name in missing:
            print(f"  - {name}")
        print()
        print(f"Total missing: {len(missing)} of {converted_total} converted tests")
    else:
        print(
            f"All {converted_total} converted tests from {log_path} are present in {yaml_path}."
        )

    if skipped:
        print("Skipped entries without '-single_device-' token:")
        for name in skipped:
            print(f"  - {name}")
        print()

    return len(missing), missing


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        dest="log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to missing_tests.log",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help="Path to test_config_inference_data_parallel.yaml",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Insert missing tests under the 'TO BE TESTED' section of the config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    missing_count, missing_tests = check_presence(args.log, args.config)

    if args.apply and missing_tests:
        insert_tests(args.config, missing_tests)
        print(f"Inserted {len(missing_tests)} tests into {args.config}")

    return missing_count


if __name__ == "__main__":
    raise SystemExit(main())

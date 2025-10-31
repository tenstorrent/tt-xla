#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Report single-device EXPECTED_PASSING tests that are not covered in data-parallel configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PARALLEL_PATH = (
    PROJECT_ROOT / "tests/runner/test_config/test_config_inference_data_parallel.yaml"
)
DEFAULT_SINGLE_DEVICE_PATH = (
    PROJECT_ROOT / "tests/runner/test_config/test_config_inference_single_device.yaml"
)


def load_yaml(path: Path) -> CommentedMap:
    yaml = YAML()
    data = yaml.load(path.read_text())
    if not isinstance(data, CommentedMap):
        raise ValueError(f"Unexpected YAML root structure in {path}")
    return data


def derive_data_parallel_name(single_device_name: str) -> str | None:
    token = "-single_device-"
    if token not in single_device_name:
        return None
    return single_device_name.replace(token, "-data_parallel-", 1)


def normalize(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def status_is_expected_passing(entry: CommentedMap | None) -> bool:
    if entry is None:
        return False
    return normalize(entry.get("status")) == "expected_passing"


def bringup_is_clean(entry: CommentedMap | None) -> bool:
    if entry is None:
        return False
    bringup = entry.get("bringup_status")
    if bringup is None:
        return True
    return normalize(bringup) == "correct_result"


def iter_single_device(entries: CommentedMap) -> Iterable[tuple[str, CommentedMap]]:
    for name, entry in entries.items():
        if not isinstance(entry, CommentedMap):
            continue
        if status_is_expected_passing(entry) and bringup_is_clean(entry):
            yield name, entry


def report_missing(
    single_device_path: Path,
    data_parallel_path: Path,
) -> int:
    single_device_root = load_yaml(single_device_path)
    data_parallel_root = load_yaml(data_parallel_path)

    sd_tests = single_device_root.get("test_config")
    dp_tests = data_parallel_root.get("test_config")

    if not isinstance(sd_tests, CommentedMap) or not isinstance(dp_tests, CommentedMap):
        raise ValueError(
            "Missing top-level 'test_config' mapping in one of the YAML files"
        )

    missing = 0
    bringup_mismatch = 0

    for sd_name, sd_entry in iter_single_device(sd_tests):
        dp_name = derive_data_parallel_name(sd_name)
        if not dp_name:
            continue

        dp_entry = dp_tests.get(dp_name)

        if not isinstance(dp_entry, CommentedMap):
            missing += 1
            print("Missing data-parallel entry:", sd_name, sep="\n  ")
            print()
            continue

        if not status_is_expected_passing(dp_entry):
            missing += 1
            print("Data-parallel not EXPECTED_PASSING:", sd_name, sep="\n  ")
            print("  data_parallel status:", dp_entry.get("status", "<missing>"))
            print()
            continue

        if not bringup_is_clean(dp_entry):
            bringup_mismatch += 1
            print("Bringup mismatch in data-parallel:", sd_name, sep="\n  ")
            print(
                "  data_parallel bringup_status:",
                dp_entry.get("bringup_status", "<missing>"),
            )
            print()

    if missing == 0:
        print(
            "All clean EXPECTED_PASSING single-device tests have matching data-parallel coverage."
        )

    print(f"Missing data-parallel coverage count: {missing}")
    if bringup_mismatch:
        print(f"Entries skipped due to bringup mismatch: {bringup_mismatch}")

    return missing


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-device",
        dest="single_device",
        type=Path,
        default=DEFAULT_SINGLE_DEVICE_PATH,
        help="Path to test_config_inference_single_device.yaml",
    )
    parser.add_argument(
        "--data-parallel",
        dest="data_parallel",
        type=Path,
        default=DEFAULT_DATA_PARALLEL_PATH,
        help="Path to test_config_inference_data_parallel.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    return report_missing(args.single_device, args.data_parallel)


if __name__ == "__main__":
    raise SystemExit(main())

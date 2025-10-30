#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Report mismatched required_pcc values between data-parallel and single-device configs."""

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


def derive_single_device_name(data_parallel_name: str) -> str | None:
    token = "-data_parallel-"
    if token not in data_parallel_name:
        return None
    return data_parallel_name.replace(token, "-single_device-", 1)


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


def get_required_pcc(entry: CommentedMap | None, default: float = 0.99) -> float:
    if entry is None:
        return default
    value = entry.get("required_pcc")
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unable to parse required_pcc value '{value}' for entry"
        ) from exc


def iter_expected_passing(entries: CommentedMap) -> Iterable[tuple[str, CommentedMap]]:
    for name, entry in entries.items():
        if not isinstance(entry, CommentedMap):
            continue
        if status_is_expected_passing(entry) and bringup_is_clean(entry):
            yield name, entry


def compare_required_pcc(
    data_parallel_path: Path,
    single_device_path: Path,
) -> int:
    data_parallel_root = load_yaml(data_parallel_path)
    single_device_root = load_yaml(single_device_path)

    dp_tests = data_parallel_root.get("test_config")
    sd_tests = single_device_root.get("test_config")

    if not isinstance(dp_tests, CommentedMap) or not isinstance(sd_tests, CommentedMap):
        raise ValueError(
            "Missing top-level 'test_config' mapping in one of the YAML files"
        )

    mismatches = 0
    missing_counterparts: list[str] = []

    for dp_name, dp_entry in iter_expected_passing(dp_tests):
        sd_name = derive_single_device_name(dp_name)
        if not sd_name:
            continue

        sd_entry = sd_tests.get(sd_name)
        if not isinstance(sd_entry, CommentedMap):
            missing_counterparts.append(sd_name)
            continue

        if not status_is_expected_passing(sd_entry) or not bringup_is_clean(sd_entry):
            continue

        dp_pcc = get_required_pcc(dp_entry)
        sd_pcc = get_required_pcc(sd_entry)

        if abs(dp_pcc - sd_pcc) > 1e-9:
            mismatches += 1
            print(
                "Mismatch:",
                dp_name,
                f"data_parallel required_pcc={dp_pcc:.3f}",
                f"single_device required_pcc={sd_pcc:.3f}",
                sep="\n  ",
            )
            print()

    if missing_counterparts:
        print("Missing single-device entries:")
        for name in missing_counterparts:
            print("  -", name)
        print()

    if mismatches == 0:
        print(
            "No required_pcc mismatches detected for EXPECTED_PASSING tests without bringup issues."
        )

    return mismatches


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-parallel",
        dest="data_parallel",
        type=Path,
        default=DEFAULT_DATA_PARALLEL_PATH,
        help="Path to test_config_inference_data_parallel.yaml",
    )
    parser.add_argument(
        "--single-device",
        dest="single_device",
        type=Path,
        default=DEFAULT_SINGLE_DEVICE_PATH,
        help="Path to test_config_inference_single_device.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    return compare_required_pcc(args.data_parallel, args.single_device)


if __name__ == "__main__":
    raise SystemExit(main())

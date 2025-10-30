#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Populate weekly markers in single-device inference configs.

This helper looks at the data-parallel inference configuration file and, for
each test listed there, finds the corresponding single-device test entry. When
found, it ensures that the single-device entry has a ``markers`` list that
includes ``"weekly"``. Existing marker lists are preserved and extended as
needed. Missing counterparts are reported to stdout so they can be reviewed.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PARALLEL_PATH = (
    PROJECT_ROOT / "tests/runner/test_config/test_config_inference_data_parallel.yaml"
)
DEFAULT_SINGLE_DEVICE_PATH = (
    PROJECT_ROOT / "tests/runner/test_config/test_config_inference_single_device.yaml"
)


def derive_single_device_name(test_name: str) -> str | None:
    """Convert a data-parallel test id into its single-device counterpart.

    Returns ``None`` when the naming convention is not recognized.
    """

    token = "-data_parallel-"
    if token not in test_name:
        return None
    return test_name.replace(token, "-single_device-", 1)


def should_add_weekly_marker(dp_entry: object) -> bool:
    if not isinstance(dp_entry, CommentedMap):
        return False

    status = dp_entry.get("status")
    if status is None:
        return False

    status_str = str(status).lower()
    return status_str == "expected_passing"


def is_incorrect_result(entry: CommentedMap | None) -> bool:
    if entry is None:
        return False
    bringup_status = entry.get("bringup_status")
    if bringup_status is None:
        return False
    return str(bringup_status).lower() == "incorrect_result"


CALCULATED_PCC_PATTERN = re.compile(r"Calculated:\s*pcc=([0-9eE+\-.]+)")
URL_PATTERN = re.compile(r"https?://\S+")


def build_incident_message(
    dp_name: str, dp_entry: CommentedMap, sd_entry: CommentedMap
) -> str:
    lines = [
        f"{dp_name}",
        "  data_parallel bringup_status: incorrect_result",
    ]

    reason = dp_entry.get("reason")
    comment_text = (
        extract_comment(dp_entry, "bringup_status", "status") if not reason else None
    )

    source_text = reason or comment_text
    calculated_pcc = extract_calculated_pcc(source_text)
    lines.append(f"  calculated_pcc: {calculated_pcc if calculated_pcc else 'N/A'}")

    if reason:
        reason_link = extract_first_url(reason)
        if reason_link:
            lines.append(f"  reason: {reason_link}")

    required_pcc = sd_entry.get("required_pcc")
    if required_pcc is not None:
        lines.append(f"  single_device required_pcc: {required_pcc}")

    return "\n".join(lines)


def build_pcc_gap_message(
    dp_name: str,
    dp_entry: CommentedMap,
    sd_entry: CommentedMap,
    dp_pcc: float,
    sd_pcc: float,
) -> str:
    lines = [
        f"{dp_name}",
        "  single_device bringup_status: incorrect_result",
        f"  data_parallel calculated_pcc: {format_pcc(dp_pcc)}",
        f"  single_device calculated_pcc: {format_pcc(sd_pcc)}",
        f"  delta: {format_pcc(sd_pcc - dp_pcc)}",
    ]

    dp_reason_link = extract_first_url(dp_entry.get("reason"))
    if dp_reason_link:
        lines.append(f"  data_parallel reason: {dp_reason_link}")

    sd_reason_link = extract_first_url(sd_entry.get("reason"))
    if sd_reason_link:
        lines.append(f"  single_device reason: {sd_reason_link}")

    return "\n".join(lines)


def build_status_incident_message(
    dp_name: str,
    dp_entry: CommentedMap,
    sd_entry: CommentedMap,
    dp_pcc: float | None,
    sd_pcc: float | None,
) -> str:
    dp_status = dp_entry.get("status")
    sd_status = sd_entry.get("status")
    sd_bringup = sd_entry.get("bringup_status")

    lines = [
        f"{dp_name}",
        f"  data_parallel status: {dp_status if dp_status is not None else 'N/A'}",
        f"  single_device status: {sd_status if sd_status is not None else 'N/A'}",
    ]

    if sd_bringup is not None:
        lines.append(f"  single_device bringup_status: {sd_bringup}")

    if dp_pcc is not None:
        lines.append(f"  data_parallel calculated_pcc: {format_pcc(dp_pcc)}")

    if sd_pcc is not None:
        lines.append(f"  single_device calculated_pcc: {format_pcc(sd_pcc)}")

    sd_reason_link = extract_first_url(sd_entry.get("reason"))
    if sd_reason_link:
        lines.append(f"  single_device reason: {sd_reason_link}")

    return "\n".join(lines)


def normalize_markers(value: object) -> CommentedSeq:
    """Return a flow-style ``CommentedSeq`` for markers."""

    if isinstance(value, CommentedSeq):
        seq = value
    elif value is None:
        seq = CommentedSeq()
    elif isinstance(value, (list, tuple, set)):
        seq = CommentedSeq(list(value))
    else:
        seq = CommentedSeq([str(value)])

    seq.fa.set_flow_style()
    return seq


def extract_comment(entry: CommentedMap, *keys: str) -> str | None:
    for key in keys:
        meta = entry.ca.items.get(key)
        if not meta:
            continue

        for comment_field in meta:
            text = comment_field_to_text(comment_field)
            if text:
                return text

    return None


def comment_field_to_text(field: object) -> str | None:
    if field is None:
        return None

    if isinstance(field, list):
        parts = [comment_field_to_text(item) for item in field]
        parts = [part for part in parts if part]
        if parts:
            return " ".join(parts)
        return None

    value = getattr(field, "value", "")
    value = value.strip()
    if value.startswith("#"):
        value = value[1:].strip()
    return value or None


def extract_calculated_pcc(text: str | None) -> str | None:
    if not text:
        return None

    match = CALCULATED_PCC_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def extract_first_url(text: str | None) -> str | None:
    if not text:
        return None

    match = URL_PATTERN.search(text)
    if not match:
        return None

    url = match.group(0)
    return url.rstrip(".,)") or None


def get_status(entry: CommentedMap | None) -> str | None:
    if entry is None:
        return None

    status = entry.get("status")
    if status is None:
        return None
    return str(status)


def normalize_status(status: str | None) -> str | None:
    if status is None:
        return None
    return status.lower()


def get_calculated_pcc_value(entry: CommentedMap | None) -> float | None:
    if entry is None:
        return None

    sources = [entry.get("reason")]
    if not sources[0]:
        sources.append(extract_comment(entry, "bringup_status", "status"))

    for source in sources:
        if not source:
            continue
        value = extract_calculated_pcc(source)
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def format_pcc(value: float) -> str:
    return f"{value:.3f}"


def ensure_weekly_marker(entry: CommentedMap) -> bool:
    """Ensure the provided entry contains ``markers: ["weekly"]``.

    Returns ``True`` if the entry was modified, ``False`` otherwise.
    """

    keys = list(entry.keys())
    markers_exists = "markers" in entry
    markers_at_end = markers_exists and keys and keys[-1] == "markers"

    existing_markers = entry.get("markers")
    seq = normalize_markers(existing_markers)

    changed = False

    weekly_indices = [idx for idx, value in enumerate(seq) if value == "weekly"]

    if weekly_indices:
        last_index = weekly_indices[-1]
        if last_index != len(seq) - 1:
            value = seq.pop(last_index)
            seq.append(value)
            changed = True
    else:
        seq.append(DoubleQuotedScalarString("weekly"))
        changed = True

    if not markers_exists:
        entry.insert(len(entry), "markers", seq)
        changed = True
    else:
        entry["markers"] = seq
        if not markers_at_end:
            entry.pop("markers")
            entry.insert(len(entry), "markers", seq)
            changed = True

    prev_key: str | None = None
    for key in entry.keys():
        if key == "markers":
            break
        prev_key = key

    if prev_key is not None:
        entry.yaml_set_comment_before_after_key(prev_key, before=None, after=None)

    entry.ca.items.pop("markers", None)
    entry.yaml_set_comment_before_after_key("markers", before=None, after=None)

    return changed


def update_single_device_config(
    data_parallel_path: Path,
    single_device_path: Path,
    dry_run: bool = False,
    skip_pcc_diff: bool = False,
) -> int:
    """Apply weekly markers to matching single-device test definitions."""

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096

    data_parallel = yaml.load(data_parallel_path.read_text())
    single_device = yaml.load(single_device_path.read_text())

    dp_tests: CommentedMap = data_parallel.get("test_config", CommentedMap())
    sd_tests: CommentedMap = single_device.get("test_config", CommentedMap())

    if not isinstance(dp_tests, CommentedMap) or not isinstance(sd_tests, CommentedMap):
        raise ValueError(
            "Unexpected YAML structure: missing top-level 'test_config' mappings"
        )

    updated = 0
    mismatch_incidents: list[str] = []
    status_incidents: list[str] = []
    pcc_gap_incidents: list[str] = []
    missing = []
    skipped = []

    for dp_name, dp_entry in dp_tests.items():
        sd_name = derive_single_device_name(dp_name)
        if not sd_name:
            skipped.append(dp_name)
            continue

        entry = sd_tests.get(sd_name)
        if entry is None:
            missing.append(sd_name)
            continue

        if not isinstance(entry, CommentedMap):
            skipped.append(dp_name)
            continue

        if not should_add_weekly_marker(dp_entry):
            continue

        dp_pcc_value = get_calculated_pcc_value(dp_entry)
        sd_pcc_value = get_calculated_pcc_value(entry)

        dp_incorrect = is_incorrect_result(dp_entry)
        sd_incorrect = is_incorrect_result(entry)

        if skip_pcc_diff and dp_incorrect and not sd_incorrect:
            mismatch_incidents.append(build_incident_message(dp_name, dp_entry, entry))
            continue

        sd_status_value = get_status(entry)
        sd_status_expected = normalize_status(sd_status_value) == "expected_passing"

        if skip_pcc_diff and (
            not sd_status_expected or (sd_incorrect and not dp_incorrect)
        ):
            status_incidents.append(
                build_status_incident_message(
                    dp_name, dp_entry, entry, dp_pcc_value, sd_pcc_value
                )
            )
            continue

        if (
            skip_pcc_diff
            and sd_incorrect
            and dp_pcc_value is not None
            and sd_pcc_value is not None
            and sd_pcc_value >= dp_pcc_value + 0.05
        ):
            pcc_gap_incidents.append(
                build_pcc_gap_message(
                    dp_name, dp_entry, entry, dp_pcc_value, sd_pcc_value
                )
            )
            continue

        if ensure_weekly_marker(entry):
            updated += 1

    stream = io.StringIO()
    yaml.dump(single_device, stream)
    rendered = stream.getvalue()
    compact_rendered = re.sub(
        r"\n{2}([ ]+markers:[^\n]*\n)",
        r"\n\1\n",
        rendered,
    )

    if dry_run:
        print("[DRY RUN] Would update", updated, "entries")
    else:
        single_device_path.write_text(compact_rendered)

    if missing:
        print("Missing single-device counterparts:")
        for name in missing:
            print("  -", name)

    if skipped:
        print("Skipped tests without recognizable counterparts:")
        for name in skipped:
            print("  -", name)

    if status_incidents:
        print("Skipped due to single_device status/bringup:")
        for message in status_incidents:
            print(message)

    if mismatch_incidents:
        print("Skipped due to incorrect_result mismatch:")
        for message in mismatch_incidents:
            print(message)

    if pcc_gap_incidents:
        print("Skipped due to PCC gap (single_device incorrect_result):")
        for message in pcc_gap_incidents:
            print(message)

    print("Updated entries:", updated)
    return updated


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-parallel",
        dest="data_parallel",
        type=Path,
        default=DEFAULT_DATA_PARALLEL_PATH,
        help="Path to the data-parallel inference YAML file",
    )
    parser.add_argument(
        "--single-device",
        dest="single_device",
        type=Path,
        default=DEFAULT_SINGLE_DEVICE_PATH,
        help="Path to the single-device inference YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many entries would be updated without writing changes",
    )
    parser.add_argument(
        "--skip-pcc-diff",
        action="store_true",
        help="Skip adding weekly markers when data-parallel is INCORRECT_RESULT but single-device is not",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    return update_single_device_config(
        data_parallel_path=args.data_parallel,
        single_device_path=args.single_device,
        dry_run=args.dry_run,
        skip_pcc_diff=args.skip_pcc_diff,
    )


if __name__ == "__main__":
    raise SystemExit(main())

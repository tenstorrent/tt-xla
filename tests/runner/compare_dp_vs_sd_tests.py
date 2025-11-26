#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare data-parallel test config against single-device equivalents.

Logs differences in `status` and `assert_pcc` between
`test_config_inference_data_parallel.yaml` and
`test_config_inference_single_device.yaml` (PyTorch).

Equivalence rule: replace the substring `-data_parallel-` with `-single_device-`
within the test id key.

Usage:
  python tests/runner/compare_dp_vs_sd_tests.py \
      --dp tests/runner/test_config/torch/test_config_inference_data_parallel.yaml \
      --sd tests/runner/test_config/torch/test_config_inference_single_device.yaml \
      [--output diff_report.txt] [--json json_report.json]

Exit code is 0 always; this is a reporting script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Fallback simple parser below


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        text = f.read()
    if yaml:
        return yaml.safe_load(text)
    # Minimal fallback: parse top-level 'test_config:' then entries by indentation.
    data: Dict[str, Any] = {}
    current_key = None
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            if line.strip() == "test_config:":
                continue
            # top-level keys not expected besides test_config
        elif line.startswith("  ") and not line.strip().startswith("#"):
            # model key or field
            if line.strip().endswith(":") and not ": " in line:
                current_key = line.strip()[:-1]
                data.setdefault(current_key, {})
            else:
                if current_key is None:
                    continue
                if ":" in line:
                    field, val = line.split(":", 1)
                    field = field.strip()
                    val = val.strip()
                    # Remove quotes
                    if (val.startswith('"') and val.endswith('"')) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    # Basic bool
                    if val.lower() in {"true", "false"}:
                        val_cast: Any = val.lower() == "true"
                    else:
                        val_cast = val
                    data[current_key][field] = val_cast
    return {"test_config": data}


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry or {}


def compute_diffs(
    dp_cfg: Dict[str, Any], sd_cfg: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    diffs: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dp_key, dp_entry in dp_cfg.items():
        if not dp_key.endswith("-data_parallel-full-inference"):
            # Skip any non data-parallel entries (should not occur)
            continue
        sd_key = dp_key.replace("-data_parallel-", "-single_device-")
        sd_entry = sd_cfg.get(sd_key)
        if sd_entry is None:
            missing.append(sd_key)
            continue
        dp_e = _normalize_entry(dp_entry)
        sd_e = _normalize_entry(sd_entry)
        dp_status = dp_e.get("status")
        sd_status = sd_e.get("status")

        # assert_pcc default True if absent; treat absence as True for comparison
        def _assert_val(e: Dict[str, Any]):
            if "assert_pcc" in e:
                return bool(e.get("assert_pcc"))
            # If required_pcc present we assume assertion enabled
            if "required_pcc" in e:
                return True
            return True  # default

        dp_assert = _assert_val(dp_e)
        sd_assert = _assert_val(sd_e)
        status_diff = dp_status != sd_status
        assert_diff = dp_assert != sd_assert
        if status_diff or assert_diff:
            diffs.append(
                {
                    "data_parallel_key": dp_key,
                    "single_device_key": sd_key,
                    "data_parallel_status": dp_status,
                    "single_device_status": sd_status,
                    "data_parallel_assert_pcc": dp_assert,
                    "single_device_assert_pcc": sd_assert,
                    "status_diff": status_diff,
                    "assert_pcc_diff": assert_diff,
                }
            )
    return diffs, missing


def format_report(diffs: List[Dict[str, Any]], missing: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"Total data-parallel entries compared: {len(diffs) + len(missing)}")
    lines.append(f"Entries with differences: {len(diffs)}")
    lines.append("")
    if missing:
        lines.append("Missing single-device counterparts:")
        for m in missing:
            lines.append(f"  - {m}")
        lines.append("")
    if diffs:
        lines.append("Differences (status / assert_pcc):")
        for d in diffs:
            lines.append(f"* {d['data_parallel_key']} -> {d['single_device_key']}")
            if d["status_diff"]:
                lines.append(
                    f"    status: DP={d['data_parallel_status']} SD={d['single_device_status']}"
                )
            if d["assert_pcc_diff"]:
                lines.append(
                    f"    assert_pcc: DP={d['data_parallel_assert_pcc']} SD={d['single_device_assert_pcc']}"
                )
    else:
        lines.append("No differences found.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Compare data-parallel vs single-device test config (PyTorch)"
    )
    ap.add_argument(
        "--dp",
        default="tests/runner/test_config/torch/test_config_inference_data_parallel.yaml",
        help="Path to data-parallel YAML",
    )
    ap.add_argument(
        "--sd",
        default="tests/runner/test_config/torch/test_config_inference_single_device.yaml",
        help="Path to single-device YAML",
    )
    ap.add_argument("--output", help="Write text report to file")
    ap.add_argument("--json", help="Write JSON diff to file")
    args = ap.parse_args()

    try:
        dp_raw = _load_yaml(args.dp)
        sd_raw = _load_yaml(args.sd)
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        sys.exit(1)

    dp_cfg = dp_raw.get("test_config", {})
    sd_cfg = sd_raw.get("test_config", {})

    diffs, missing = compute_diffs(dp_cfg, sd_cfg)
    report = format_report(diffs, missing)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
    if args.json:
        json_obj = {"differences": diffs, "missing_single_device": missing}
        with open(args.json, "w") as f:
            json.dump(json_obj, f, indent=2)


if __name__ == "__main__":
    main()

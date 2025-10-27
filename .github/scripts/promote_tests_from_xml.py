#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception as e:
    print(
        "PyYAML is required to run this script. Please install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class TestResult:
    key: (
        str  # e.g., llama/causal_lm/pytorch-llama_3_1_8b-tensor_parallel-full-inference
    )
    bringup_status: str  # PASSED, INCORRECT_RESULT, FAILED_RUNTIME, ...
    pcc_value: Optional[float]
    pcc_threshold: Optional[float]
    pcc_assertion_enabled: Optional[bool]
    arch: Optional[str]
    model_group: Optional[str]


PromotionPlan = Dict[str, Dict[str, Dict[str, object]]]
# plan[config_file][test_key] = { 'action': 'add'|'update'| 'noop', 'new_entry': dict, 'old_entry': dict|None, 'reason': str }


# -----------------------------
# Helpers
# -----------------------------


def load_xml_files(patterns: Iterable[str]) -> List[str]:
    matched_files: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path not in seen:
                matched_files.append(path)
                seen.add(path)
        if not glob.has_magic(pattern) and os.path.exists(pattern):
            if pattern not in seen:
                matched_files.append(pattern)
                seen.add(pattern)
    return matched_files


def extract_bracket_key_from_testcase_name(name: str) -> Optional[str]:
    # Expect: test_all_models[<key>]
    start = name.find("[")
    end = name.rfind("]")
    if start == -1 or end == -1 or end <= start + 1:
        return None
    return name[start + 1 : end]


def parse_testcase_properties(
    properties_elem: ET.Element,
) -> Tuple[Dict[str, object], Optional[str], Optional[str]]:
    """Return (tags_dict, group, owner)."""
    tags_dict: Dict[str, object] = {}
    group_value: Optional[str] = None
    owner_value: Optional[str] = None
    for prop in properties_elem:
        name = prop.get("name")
        if name == "tags":
            raw = prop.get("value", "{}")
            try:
                tags_dict = ast.literal_eval(raw)
            except Exception:
                tags_dict = {}
        elif name == "group":
            group_value = prop.get("value")
        elif name == "owner":
            owner_value = prop.get("value")
    return tags_dict, group_value, owner_value


def parse_junit_xml(xml_file: str) -> Dict[str, TestResult]:
    results: Dict[str, TestResult] = {}
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}", file=sys.stderr)
        return results
    except FileNotFoundError:
        print(f"File not found: {xml_file}", file=sys.stderr)
        return results

    for testcase in root.findall(".//testcase"):
        name = testcase.get("name", "")
        key = extract_bracket_key_from_testcase_name(name)
        if not key:
            # Not a parametrized test we care about
            continue

        properties = testcase.find("properties")
        if properties is None:
            continue
        tags, group_value, _ = parse_testcase_properties(properties)

        bringup_status = str(tags.get("bringup_status", ""))
        if not bringup_status:
            # No bringup status reported; skip
            continue

        # Pull additional signal for future decisions
        pcc_value = tags.get("pcc")
        pcc_threshold = tags.get("pcc_threshold")
        pcc_assertion_enabled = tags.get("pcc_assertion_enabled")
        arch = tags.get("arch")

        # Normalize optional floats/bools
        try:
            pcc_value = float(pcc_value) if pcc_value is not None else None
        except Exception:
            pcc_value = None
        try:
            pcc_threshold = float(pcc_threshold) if pcc_threshold is not None else None
        except Exception:
            pcc_threshold = None
        if isinstance(pcc_assertion_enabled, str):
            pcc_assertion_enabled = pcc_assertion_enabled.lower() == "true"

        results[key] = TestResult(
            key=key,
            bringup_status=str(bringup_status),
            pcc_value=pcc_value,
            pcc_threshold=pcc_threshold,
            pcc_assertion_enabled=(
                bool(pcc_assertion_enabled)
                if pcc_assertion_enabled is not None
                else None
            ),
            arch=str(arch) if arch is not None else None,
            model_group=group_value,
        )

    return results


# -----------------------------
# Mapping test key to config file
# -----------------------------


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def map_test_to_config_file(test_key: str) -> Optional[str]:
    """Minimal mapping from test key suffix to a config YAML path."""
    suffix_to_config = {
        "-single_device-full-inference": "tests/runner/test_config/test_config_inference_single_device.yaml",
        "-tensor_parallel-full-inference": "tests/runner/test_config/test_config_inference_tensor_parallel.yaml",
        "-data_parallel-full-inference": "tests/runner/test_config/test_config_inference_data_parallel.yaml",
        "-single_device-full-training": "tests/runner/test_config/test_config_training_single_device.yaml",
    }

    for suffix, rel_path in suffix_to_config.items():
        if test_key.endswith(suffix):
            return os.path.join(REPO_ROOT, rel_path)
    return None


# -----------------------------
# YAML IO
# -----------------------------


def load_yaml_config(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to read YAML {path}: {e}", file=sys.stderr)
            return {}
    return data


def write_yaml_config(path: str, data: Dict[str, object]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# -----------------------------
# Promotion logic
# -----------------------------


def build_promotion_plan(results: Dict[str, TestResult]) -> PromotionPlan:
    plan: PromotionPlan = defaultdict(dict)

    for key, res in results.items():
        config_path = map_test_to_config_file(key)
        if not config_path:
            continue

        data = load_yaml_config(config_path)
        if "test_config" not in data or data["test_config"] is None:
            data["test_config"] = {}
        test_config: Dict[str, Dict[str, object]] = data["test_config"]  # type: ignore

        current_entry = test_config.get(key)

        junit_bringup = (res.bringup_status or "").upper()

        def make_entry_for_passed() -> Dict[str, object]:
            new_entry = dict(current_entry or {})
            # Ensure EXPECTED_PASSING status and bringup_status: PASSED
            new_entry["status"] = "EXPECTED_PASSING"
            new_entry["bringup_status"] = "PASSED"
            return new_entry

        def make_entry_for_incorrect_result() -> Dict[str, object]:
            new_entry = dict(current_entry or {})
            new_entry["status"] = "EXPECTED_PASSING"
            new_entry["assert_pcc"] = False
            new_entry["bringup_status"] = "INCORRECT_RESULT"
            return new_entry

        action = "noop"
        reason = ""
        new_entry: Optional[Dict[str, object]] = None

        if junit_bringup == "PASSED":
            if current_entry is None:
                action = "add"
                reason = "New test observed as PASSED"
                new_entry = make_entry_for_passed()
            else:
                curr_status = str(current_entry.get("status", ""))
                curr_bringup = str(current_entry.get("bringup_status", ""))
                if curr_status != "EXPECTED_PASSING" or curr_bringup != "PASSED":
                    action = "update"
                    reason = f"Promote to PASSED (was status={curr_status or 'unset'}, bringup_status={curr_bringup or 'unset'})"
                    new_entry = make_entry_for_passed()

        elif junit_bringup == "INCORRECT_RESULT":
            if current_entry is None:
                action = "add"
                reason = "New test runs end-to-end with PCC errors (INCORRECT_RESULT)"
                new_entry = make_entry_for_incorrect_result()
            else:
                curr_status = str(current_entry.get("status", ""))
                curr_bringup = str(current_entry.get("bringup_status", ""))
                # Anything not PASSED or INCORRECT_RESULT is worse; promote to INCORRECT_RESULT
                if (
                    curr_bringup != "INCORRECT_RESULT"
                    or curr_status != "EXPECTED_PASSING"
                    or current_entry.get("assert_pcc") is not False
                ):
                    action = "update"
                    reason = f"Promote to INCORRECT_RESULT with assert_pcc=false (was status={curr_status or 'unset'}, bringup_status={curr_bringup or 'unset'})"
                    new_entry = make_entry_for_incorrect_result()

        # Ignore other bringup statuses for first pass

        if action != "noop" and new_entry is not None:
            plan[config_path][key] = {
                "action": action,
                "new_entry": new_entry,
                "old_entry": current_entry,
                "reason": reason,
                "group": res.model_group,
            }

    return plan


def print_plan(plan: PromotionPlan) -> None:
    if not plan:
        print("No promotions or improvements detected.")
        return
    for config_path in sorted(plan.keys()):
        print(f"\nConfig: {config_path}")
        for key, change in plan[config_path].items():
            action = change["action"]
            reason = change["reason"]
            group = change.get("group") or "unknown"
            print(f"  - {action.upper():<6} {group:<8} {key}")
            print(f"    reason: {reason}")


def apply_plan(plan: PromotionPlan, write_files: bool) -> List[str]:
    """Apply the plan. If write_files is True, write back to the original YAML files.
    Returns list of written files.
    """
    written: List[str] = []
    for config_path, changes in plan.items():
        if not changes:
            continue
        data = load_yaml_config(config_path)
        if "test_config" not in data or data["test_config"] is None:
            data["test_config"] = {}
        test_config: Dict[str, Dict[str, object]] = data["test_config"]  # type: ignore

        for key, change in changes.items():
            test_config[key] = change["new_entry"]  # type: ignore

        if write_files:
            write_yaml_config(config_path, data)
            written.append(config_path)
            print(f"UPDATED: {config_path}")
    return written


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote tests based on JUnit XML bringup results"
    )
    parser.add_argument(
        "--xml",
        nargs="+",
        required=True,
        help="Path(s) or glob pattern(s) to JUnit XML file(s)",
    )
    parser.add_argument(
        "--promote-tests",
        action="store_true",
        help="When set, write proposed changes back to original YAML. Otherwise, dry-run only.",
    )

    args = parser.parse_args()

    xml_files = load_xml_files(args.xml)
    if not xml_files:
        print("No XML files matched the provided --xml pattern(s).", file=sys.stderr)
        sys.exit(1)

    aggregate_results: Dict[str, TestResult] = {}
    for f in xml_files:
        res = parse_junit_xml(f)
        # Merge; last writer wins for simplicity
        aggregate_results.update(res)

    plan = build_promotion_plan(aggregate_results)
    print_plan(plan)

    if args.promote_tests:
        written = apply_plan(plan, write_files=True)
        if not written:
            print("No files written (no changes).")
    else:
        # Dry-run only
        _ = apply_plan(plan, write_files=False)


if __name__ == "__main__":
    main()

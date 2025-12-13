#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import glob
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
except Exception as e:
    print("ruamel.yaml is required. Run: pip install ruamel.yaml", file=sys.stderr)
    raise

# Small buffer consistent with guidance logic elsewhere
PCC_BUFFER = 0.004


# Build and return the CLI argument parser for this tool.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote PCC settings in test_config YAMLs based on JUnit XML guidance tags"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--xml",
        nargs="+",
        help="One or more XML files, directories, or globs (e.g. ./*/*.xml). Alternative to --run-id; cannot be combined.",
    )
    group.add_argument(
        "--run-id",
        help="GitHub run-id to download report artifacts, then process them. Alternative to --xml; cannot be combined.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to YAML files (default: dry-run, print plan only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostic information while processing.",
    )
    return parser


# Resolve input sources: either --xml patterns or download artifacts for --run-id.
def resolve_input_patterns(
    xml_patterns: Optional[List[str]], run_id: Optional[str]
) -> List[str]:
    patterns: List[str] = []
    if xml_patterns:
        for p in xml_patterns:
            patterns.append(p)
        return patterns
    if run_id:
        out_dir = f"run_id_{run_id}_reports"
        existing_xmls = glob.glob(os.path.join(out_dir, "*", "*.xml"))
        if not (os.path.isdir(out_dir) and existing_xmls):
            try:
                subprocess.run(
                    [
                        sys.executable,
                        ".github/scripts/download_artifacts.py",
                        "--run-id",
                        str(run_id),
                        "--filter",
                        "report",
                        "--output",
                        out_dir,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"Artifact download failed for run-id {run_id}: {e}",
                    file=sys.stderr,
                )
                return []
        patterns.append(out_dir)
        return patterns
    return patterns


# Expand patterns into a unique, sorted list of XML file paths.
def collect_input_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            files.extend(
                glob.glob(os.path.join(pattern, "**", "*.xml"), recursive=True)
            )
        elif os.path.isfile(pattern):
            files.append(pattern)
        else:
            files.extend(glob.glob(pattern, recursive=True))
    uniq = sorted({os.path.abspath(p) for p in files})
    return [p for p in uniq if os.path.isfile(p)]


# Extract testsuite timestamp (epoch seconds) to order/dedupe results.
def get_suite_timestamp(tree: ET.ElementTree) -> Optional[float]:
    suite = tree.find(".//testsuite")
    if suite is None:
        return None
    ts = suite.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


# Safely parse the tags property (Python-literal dict stored as a string).
def parse_tags_value(value: str) -> Dict:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


# Yield test records (dict) parsed from each <testcase>'s tags (and group).
def iter_test_records(tree: ET.ElementTree) -> Iterator[Dict]:
    root = tree.getroot()
    for tc in root.findall(".//testcase"):
        props = tc.find("properties")
        if props is None:
            continue
        tags_val = None
        group_val = ""
        for prop in props.findall("property"):
            name = prop.get("name")
            if name == "tags":
                tags_val = prop.get("value")
            elif name == "group":
                group_val = prop.get("value") or ""
        if not tags_val:
            continue
        tags = parse_tags_value(tags_val)
        if not isinstance(tags, dict):
            continue
        record: Dict = dict(tags)
        record["group"] = group_val
        # Include testcase name for mapping; some JUnit formats mirror in 'specific_test_case'
        # We expect 'specific_test_case' to be present in tags as per our test infra
        yield record


# Map a testcase name to its corresponding test_config YAML path.
def map_test_to_config_file(test_name: str) -> Optional[str]:
    framework_dir = "torch" if test_name.startswith("test_all_models_torch") else "jax"
    suffix_to_filename = {
        "-single_device-full-inference": "test_config_inference_single_device.yaml",
        "-tensor_parallel-full-inference": "test_config_inference_tensor_parallel.yaml",
        "-data_parallel-full-inference": "test_config_inference_data_parallel.yaml",
        "-single_device-full-training": "test_config_training_single_device.yaml",
    }
    for suffix, fname in suffix_to_filename.items():
        if suffix in test_name:
            return os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                "tests/runner/test_config",
                framework_dir,
                fname,
            )
    return None


# Compute the next 0.01 step above a threshold (capped at 0.99).
def next_centesimal_level(value: float) -> float:
    nxt = (int(value * 100) + 1) / 100.0
    return min(0.99, nxt)


# Load a YAML test_config file into a CommentedMap (with stable formatting).
def load_yaml_config(path: str) -> CommentedMap:
    yaml = YAML(typ="rt")
    yaml.allow_duplicate_keys = False
    yaml.preserve_quotes = True
    yaml.width = 100000
    if not os.path.exists(path):
        return CommentedMap()
    with open(path, "r") as f:
        try:
            data = yaml.load(f) or CommentedMap()
        except Exception as e:
            print(f"Failed to read YAML {path}: {e}", file=sys.stderr)
            return CommentedMap()
    if not isinstance(data, CommentedMap):
        data = CommentedMap(data or {})
    return data


# Write a CommentedMap back to a YAML test_config file (preserving style).
def write_yaml_config(path: str, data: CommentedMap) -> None:
    yaml = YAML(typ="rt")
    yaml.allow_duplicate_keys = False
    yaml.preserve_quotes = True
    yaml.width = 100000
    with open(path, "w") as f:
        yaml.dump(data, f)


# Extract the parameterized key from a pytest-style testcase name.
def extract_bracket_key_from_testcase_name(name: str) -> Optional[str]:
    start = name.find("[")
    end = name.rfind("]")
    if start == -1 or end == -1 or end <= start + 1:
        return None
    return name[start + 1 : end]


# Parse XMLs and return the latest (by suite timestamp) guidance per (test, arch).
def collect_guidance_updates(
    xml_files: List[str], verbose: bool
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Return dict keyed by (test_name, arch) tuple with desired actions."""
    desired: Dict[Tuple[str, str], Dict[str, object]] = {}
    # Dedupe latest per (test_name, arch)
    latest_by_key: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
    for path in xml_files:
        try:
            tree = ET.parse(path)
        except Exception as e:
            print(f"Failed to parse XML: {path}: {e}", file=sys.stderr)
            continue
        score = get_suite_timestamp(tree) or 0.0
        for rec in iter_test_records(tree):
            test_name = str(rec.get("specific_test_case") or "")
            if not test_name.startswith("test_all_models"):
                continue
            arch_val = str(rec.get("arch") or "")
            key = (test_name, arch_val)
            prev = latest_by_key.get(key)
            if prev is None or score >= prev[0]:
                latest_by_key[key] = (score, rec)
    for (_test_name, _arch), (_score, rec) in latest_by_key.items():
        guidance = rec.get("guidance")
        if isinstance(guidance, str):
            tags = [g.strip() for g in guidance.split(",") if g.strip()]
        elif isinstance(guidance, list):
            tags = [str(x) for x in guidance if x]
        else:
            tags = []
        if not tags:
            continue
        desired[(_test_name, _arch)] = {
            "guidance": tags,
            "pcc_threshold": rec.get("pcc_threshold"),
            "pcc_value": rec.get("pcc"),
        }
        if verbose:
            print(
                f"Guidance for {_test_name}: arch: {_arch} pcc: {rec.get('pcc')} tags: {tags} (th={rec.get('pcc_threshold')})"
            )
    return desired


# Decide the minimal YAML changes for a given test based on its guidance tags.
def plan_updates_for_test(
    test_name: str, test_info: Dict[str, object]
) -> Dict[str, object]:
    """Decide the minimal YAML changes based on guidance tags."""
    tags: List[str] = list(test_info.get("guidance") or [])
    pcc_th = test_info.get("pcc_threshold")
    pcc_val = test_info.get("pcc_value")
    try:
        pcc_th_f = float(pcc_th) if pcc_th is not None else None
    except Exception:
        pcc_th_f = None
    try:
        pcc_val_f = float(pcc_val) if pcc_val is not None else None
    except Exception:
        pcc_val_f = None
    plan: Dict[str, object] = {}
    if "ENABLE_PCC_099" in tags or "ENABLE_PCC" in tags:
        # Remove assert_pcc:false (do not set True; absence implies enabled by default)
        plan["remove_assert_pcc_false"] = True
    # Raising thresholds
    elif "RAISE_PCC_099" in tags or "RAISE_PCC" in tags:
        # Set required_pcc based on achieved PCC minus buffer, then floored to centesimal, capped at 0.99
        # This ensures the new threshold is safely below the achieved PCC
        if pcc_val_f is not None:
            # Subtract buffer first, then floor to centesimal to determine safe threshold
            adjusted_pcc = pcc_val_f - PCC_BUFFER
            new_th = min(0.99, int(adjusted_pcc * 100) / 100.0)
            if pcc_th_f is None or new_th > pcc_th_f:
                plan["set_required_pcc"] = new_th
    return plan


# Apply a single test's plan to its YAML entry; optionally write back to disk.
def apply_updates_to_yaml(
    config_path: str,
    test_name: str,
    plan: Dict[str, object],
    write: bool,
    verbose: bool,
) -> None:
    data = load_yaml_config(config_path)
    if "test_config" not in data or data["test_config"] is None:
        print(f"[WARN] No test_config section in {config_path}", file=sys.stderr)
        return
    test_config = data["test_config"]  # type: ignore
    bracket_key = extract_bracket_key_from_testcase_name(test_name) or ""
    if not bracket_key or bracket_key not in test_config:
        print(
            f"[WARN] Test entry not found in {config_path} for {test_name}",
            file=sys.stderr,
        )
        return
    entry = test_config.get(bracket_key) or {}
    # Remove assert_pcc:false
    if plan.get("remove_assert_pcc_false"):
        if entry.get("assert_pcc") is False:
            if verbose:
                print(f" - Removing assert_pcc:false for {bracket_key}")
            entry.pop("assert_pcc", None)
    # Set required_pcc
    if "set_required_pcc" in plan:
        new_th = float(plan["set_required_pcc"])  # type: ignore
        old_th = entry.get("required_pcc")
        if old_th is None or float(old_th) < new_th:
            if verbose:
                print(
                    f" - Setting required_pcc: {old_th} -> {new_th} for {bracket_key}"
                )
            entry["required_pcc"] = new_th
    test_config[bracket_key] = entry  # type: ignore
    if write:
        write_yaml_config(config_path, data)


# Entry point: resolve inputs, collect guidance, plan and apply (or print) updates.
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    patterns = resolve_input_patterns(args.xml, args.run_id)
    if not patterns:
        print("No XML sources resolved. Use --xml or --run-id.", file=sys.stderr)
        return 2
    files = collect_input_files(patterns)
    if not files:
        print("No XML files matched.", file=sys.stderr)
        return 1

    start = time.time()
    print(f"Collecting guidance updates from {len(files)} files...", flush=True)
    desired = collect_guidance_updates(files, verbose=args.verbose)
    print(
        f"Collected {len(desired)} guidance updates. Took {time.time() - start:.2f} seconds.",
        flush=True,
    )

    # Debug - early exit here:
    # return 0

    if not desired:
        print("No guidance found in provided artifacts.")
        return 0

    print("Generating promotion plan now...", flush=True)
    for (test_name, arch), info in sorted(desired.items()):
        cfg = map_test_to_config_file(test_name)
        if not cfg:
            print(f"  - SKIP (no config): {test_name} [arch: {arch}]")
            continue
        plan = plan_updates_for_test(test_name, info)
        if not plan:
            print(f"  - NOOP: {test_name} [arch: {arch}]")
            continue
        print(
            f"  - {('APPLY' if args.apply else 'PLAN ')} {os.path.basename(cfg)} :: {test_name} [arch: {arch}] -> {plan}"
        )
        apply_updates_to_yaml(
            cfg, test_name, plan, write=args.apply, verbose=args.verbose
        )

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s. Mode: {'apply' if args.apply else 'dry-run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

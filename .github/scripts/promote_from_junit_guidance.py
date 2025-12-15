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
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Use test_config.testing directory instead of test_config (for dev/testing).",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip optimization pass (for debugging - leaves arch_overrides as-is).",
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
def map_test_to_config_file(test_name: str, testing: bool = False) -> Optional[str]:
    framework_dir = "torch" if test_name.startswith("test_all_models_torch") else "jax"
    config_dir = "test_config.testing" if testing else "test_config"
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
                "tests/runner",
                config_dir,
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


# Parse guidance tags from various formats (string, list, etc.)
def parse_guidance_tags(guidance: object) -> List[str]:
    """Extract guidance tags from string, list, or other formats."""
    if isinstance(guidance, str):
        return [g.strip() for g in guidance.split(",") if g.strip()]
    if isinstance(guidance, list):
        return [str(x) for x in guidance if x]
    return []


# Parse XMLs and return the latest (by suite timestamp) guidance per (test, arch).
def collect_guidance_updates(
    xml_files: List[str], verbose: bool
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Return dict keyed by (test_name, arch) tuple with desired actions."""
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
            key = (test_name, str(rec.get("arch") or ""))
            if key not in latest_by_key or score >= latest_by_key[key][0]:
                latest_by_key[key] = (score, rec)

    desired: Dict[Tuple[str, str], Dict[str, object]] = {}
    for (test_name, arch), (score, rec) in latest_by_key.items():
        tags = parse_guidance_tags(rec.get("guidance"))
        if not tags:
            continue
        desired[(test_name, arch)] = {
            "guidance": tags,
            "pcc_threshold": rec.get("pcc_threshold"),
            "pcc_value": rec.get("pcc"),
        }
        if verbose:
            print(
                f"Guidance for {test_name}: arch: {arch} pcc: {rec.get('pcc')} tags: {tags} (th={rec.get('pcc_threshold')})"
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


# Get the set of all known archs for a test entry (currently n150, p150; will be calculated later).
def get_all_archs_for_entry(entry: CommentedMap) -> set[str]:
    """Return set of all archs that exist in arch_overrides or are known defaults."""
    archs = set()
    arch_overrides = entry.get("arch_overrides") or {}
    if isinstance(arch_overrides, dict):
        archs.update(arch_overrides.keys())
    # For now, assume standard archs are n150 and p150
    # TODO: Calculate this dynamically based on test configuration
    archs.update(["n150", "p150"])
    return archs


# Get a field value from arch_overrides first, then top-level entry if not found.
def get_arch_or_top(
    arch_entry: CommentedMap, entry: CommentedMap, key: str
) -> Optional[object]:
    """Get field value from arch_entry first, fallback to entry (top-level) if not found."""
    val = arch_entry.get(key)
    return val if val is not None else entry.get(key)


# Add a key to a CommentedMap while preserving trailing comments.
def add_key_preserving_trailing_comment(
    commented_map: CommentedMap, new_key: str, new_value: object, verbose: bool = False
) -> None:
    """
    Add a key to a CommentedMap while preserving trailing comments.
    If the last key has a trailing comment (blank lines + next section comment),
    move it to the new key being added.

    For nested CommentedMaps, the trailing comment is attached to the last key
    of the innermost nested structure.
    """
    print(
        f"KCM Inside add_key_preserving_trailing_comment with new_key: {new_key} and new_value: {new_value}"
    )
    # Save trailing comment from the current last key
    trailing_comment = None
    if commented_map and hasattr(commented_map, "ca") and commented_map.ca.items:
        last_key = list(commented_map.keys())[-1]
        if last_key in commented_map.ca.items:
            comment_list = commented_map.ca.items[last_key]
            if len(comment_list) > 2 and comment_list[2]:
                trailing_comment = comment_list[2]
                # Clear it from the old position
                cleaned = list(comment_list)
                cleaned[2] = None
                commented_map.ca.items[last_key] = cleaned

    # Add the new key
    commented_map[new_key] = new_value

    # Attach the trailing comment to the appropriate location
    if trailing_comment:
        # If the value is a nested CommentedMap, attach to the innermost last key
        if isinstance(new_value, CommentedMap) and new_value:
            # Navigate to the deepest last key
            if verbose:
                print(f"Found nested CommentedMap for key: {new_key}")
            current = new_value
            path = []

            while isinstance(current, CommentedMap) and current:
                last_nested_key = list(current.keys())[-1]
                path.append((current, last_nested_key))
                last_nested_value = current[last_nested_key]

                if isinstance(last_nested_value, CommentedMap) and last_nested_value:
                    current = last_nested_value
                else:
                    # Found the innermost last key - attach comment here
                    break

            # Attach to the deepest key we found
            if path:
                target_map, target_key = path[-1]
                if target_key not in target_map.ca.items:
                    target_map.ca.items[target_key] = [None, None, None, None]
                inner_comment_list = list(
                    target_map.ca.items.get(target_key, [None, None, None, None])
                )
                while len(inner_comment_list) < 4:
                    inner_comment_list.append(None)
                inner_comment_list[2] = trailing_comment
                target_map.ca.items[target_key] = inner_comment_list
        else:
            if verbose:
                print(f"Found simple value for key: {new_key}")
            # Simple value, attach to the new key itself
            if new_key not in commented_map.ca.items:
                commented_map.ca.items[new_key] = [None, None, None, None]
            new_comment_list = list(
                commented_map.ca.items.get(new_key, [None, None, None, None])
            )
            while len(new_comment_list) < 4:
                new_comment_list.append(None)
            new_comment_list[2] = trailing_comment
            commented_map.ca.items[new_key] = new_comment_list


def move_field_preserving_comments(
    source_map: CommentedMap,
    target_map: CommentedMap,
    field: str,
    value: object,
) -> None:
    """
    Move a field from source to target, preserving all inline and trailing comments.

    This function is used during the optimization pass when moving common fields from
    arch_overrides back to the top level (e.g., when all archs have required_pcc=0.96,
    move it to top-level and remove from each arch).

    Key challenge: When the optimization pass identifies that a field should be moved
    to the top level, we need to preserve any inline comments (like "# Issue 1234")
    that were attached to that field in one of the arch override entries.

    Comment preservation strategy:
    - Index 1 (inline): Copy from source only if target doesn't have one
    - Index 2 (trailing): Copy from source only if target doesn't have one

    This prevents overwriting existing comments on the target while preserving
    valuable comment context from the source location.

    Note: The source_map is typically an arch-specific entry (e.g., arch_overrides.n150)
    and target_map is the top-level test entry. We want to preserve comments when
    consolidating duplicate values.
    """
    # Get existing comments from source
    source_comments = None
    if (
        hasattr(source_map, "ca")
        and source_map.ca.items
        and field in source_map.ca.items
    ):
        source_comments = list(source_map.ca.items[field])

    # Set value at target
    target_map[field] = value
    print(
        f"KCM move_field_preserving_comments starting with source_map: {source_map} and target_map: {target_map} and field: {field} and value: {value} source_comments: {source_comments}"
    )

    # If source had comments, transfer them to target
    if source_comments:
        print(f"KCM - Source had comments for field: {field}")
        # Get existing target comments if any
        target_comments = [None, None, None, None]
        print(f"KCM - Target had comments for field: {field}")
        if (
            hasattr(target_map, "ca")
            and target_map.ca.items
            and field in target_map.ca.items
        ):
            target_comments = list(
                target_map.ca.items.get(field, [None, None, None, None])
            )

        while len(target_comments) < 4:
            target_comments.append(None)
        while len(source_comments) < 4:
            source_comments.append(None)

        # Copy inline comment (index 1) from source if target doesn't have one
        if source_comments[1] and not target_comments[1]:
            print(
                f"KCM - Copying inline comment from source to target for field: {field} source_comments[1]: {source_comments[1]} target_comments[1]: {target_comments[1]}"
            )
            target_comments[1] = source_comments[1]

        # Copy trailing comment (index 2) from source if target doesn't have one
        if source_comments[2] and not target_comments[2]:
            print(
                f"KCM - Copying trailing comment from source to target for field: {field} source_comments[2]: {source_comments[2]} target_comments[2]: {target_comments[2]}"
            )
            target_comments[2] = source_comments[2]

        target_map.ca.items[field] = target_comments


def apply_updates_to_yaml(
    data: CommentedMap,
    test_name: str,
    arch_plans: Dict[str, Dict[str, object]],
    verbose: bool,
) -> Optional[str]:
    """
    Apply changes grouped by arch, always using arch_overrides.
    Returns the bracket_key if modifications were made, None otherwise.
    """
    if "test_config" not in data or data["test_config"] is None:
        return None
    test_config = data["test_config"]  # type: ignore
    bracket_key = extract_bracket_key_from_testcase_name(test_name) or ""
    if not bracket_key or bracket_key not in test_config:
        return None
    entry = test_config.get(bracket_key) or {}
    if not isinstance(entry, CommentedMap):
        entry = CommentedMap(entry)
        test_config[bracket_key] = entry

    # Check if arch_overrides already exists
    arch_overrides_existed = "arch_overrides" in entry

    # Reuse existing arch_overrides or create new one (DON'T copy!)
    if arch_overrides_existed:
        arch_overrides = entry.get("arch_overrides")
        if not isinstance(arch_overrides, CommentedMap):
            # Convert plain dict to CommentedMap
            arch_overrides = CommentedMap(arch_overrides)
            entry["arch_overrides"] = arch_overrides
    else:
        arch_overrides = CommentedMap()

    # Track if we made any modifications
    modified = False

    # Apply changes per arch
    for arch, plan in arch_plans.items():
        # Reuse existing arch_entry or create new one (DON'T copy!)
        arch_entry = arch_overrides.get(arch)
        if arch_entry is None:
            arch_entry = CommentedMap()
            arch_overrides[arch] = arch_entry
        elif not isinstance(arch_entry, CommentedMap):
            # Convert plain dict to CommentedMap while preserving structure
            arch_entry = CommentedMap(arch_entry)
            arch_overrides[arch] = arch_entry
        # else: arch_entry is already a CommentedMap, reuse it as-is

        # Apply required_pcc change
        if "set_required_pcc" in plan:
            new_th = float(plan["set_required_pcc"])  # type: ignore
            old_th = get_arch_or_top(arch_entry, entry, "required_pcc")
            if old_th is None or float(old_th) < new_th:
                if verbose:
                    print(
                        f"   - Setting arch_overrides.{arch}.required_pcc: {old_th} -> {new_th} for {bracket_key}"
                    )
                arch_entry["required_pcc"] = new_th
                modified = True

        # Apply assert_pcc change (remove assert_pcc:false)
        if plan.get("remove_assert_pcc_false"):
            assert_pcc_val = get_arch_or_top(arch_entry, entry, "assert_pcc")
            if assert_pcc_val is False:
                if verbose:
                    print(
                        f"   - Removing arch_overrides.{arch}.assert_pcc:false for {bracket_key}"
                    )
                arch_entry.pop("assert_pcc", None)
                modified = True
                # Remove empty arch entry
                if not arch_entry:
                    arch_overrides.pop(arch, None)

    # Add arch_overrides if we're creating it for the first time
    if modified and not arch_overrides_existed:
        if arch_overrides and len(arch_overrides) > 0:
            add_key_preserving_trailing_comment(
                entry, "arch_overrides", arch_overrides, verbose
            )

    # Clean up empty arch_overrides
    if "arch_overrides" in entry and (not arch_overrides or len(arch_overrides) == 0):
        entry.pop("arch_overrides", None)

    return bracket_key if modified else None


# Optimize arch_overrides by moving common fields to top-level and cleaning up.
def optimize_arch_overrides(
    entry: CommentedMap, all_archs: set[str], verbose: bool, bracket_key: str
) -> None:
    """
    Generic optimization pass: move fields to top-level if common across all archs.
    This is field-agnostic and works for any field in arch_overrides.
    """
    arch_overrides = entry.get("arch_overrides")
    if not arch_overrides or not isinstance(arch_overrides, dict):
        return

    # Collect all fields that appear in any arch override
    all_fields: set[str] = set()
    for arch_entry in arch_overrides.values():
        if isinstance(arch_entry, dict):
            all_fields.update(arch_entry.keys())

    # Collect fields to move to top-level (common across all archs)
    fields_to_move: Dict[str, Tuple[object, Optional[CommentedMap]]] = {}
    for field in all_fields:
        field_values: Dict[str, object] = {}
        # Track which arch has the field WITH COMMENTS (to preserve them)
        source_arch_map = None

        # Iterate through arch_overrides in YAML order to find the LAST arch with this field
        # This is important because the last arch will have the trailing comment which is
        # occasionally a line break between tests.
        if isinstance(arch_overrides, dict):
            for arch in arch_overrides.keys():  # Preserves YAML order
                arch_entry = arch_overrides.get(arch)
                if isinstance(arch_entry, dict) and field in arch_entry:
                    field_values[arch] = arch_entry[field]
                    # Always update source_arch_map - we want the LAST arch in YAML order
                    if isinstance(arch_entry, CommentedMap):
                        source_arch_map = arch_entry

        # Now check if any remaining archs (not in arch_overrides) inherit from top-level
        for arch in all_archs:
            if arch not in field_values:
                if field in entry:
                    # Arch doesn't have override, use top-level value
                    field_values[arch] = entry[field]

        # If all archs have the same value (or inherit same top-level), mark for moving
        if len(field_values) == len(all_archs):
            values = list(field_values.values())
            first_value = values[0]
            if all(v == first_value for v in values):
                # Only move if top-level doesn't already have it, or if it's different
                if field not in entry or entry[field] != first_value:
                    fields_to_move[field] = (first_value, source_arch_map)

    # Move fields to top-level and remove from arch_overrides
    if fields_to_move:
        for field, (common_value, source_map) in fields_to_move.items():
            if verbose:
                print(
                    f" - Optimizing: moving {field}={common_value} to top-level (common across all archs) for {bracket_key}"
                )

            # Move the field with its comments from source to target
            if source_map:
                print(
                    f"KCM - Moving field {field} from source_map to entry with source_map: {source_map} and entry: {entry}"
                )
                move_field_preserving_comments(source_map, entry, field, common_value)
            else:
                print(
                    f"KCM - Moving field {field} from entry to entry with entry: {entry}"
                )
                entry[field] = common_value

            # Remove from all arch_overrides
            for arch in all_archs:
                arch_entry = arch_overrides.get(arch)
                if isinstance(arch_entry, dict) and field in arch_entry:
                    arch_entry.pop(field, None)
                    # Remove empty arch entry
                    if not arch_entry:
                        arch_overrides.pop(arch, None)

        # Reorder: move arch_overrides to end so top-level fields appear first
        if "arch_overrides" in entry and entry["arch_overrides"]:
            arch_overrides_val = entry.pop("arch_overrides")
            entry["arch_overrides"] = arch_overrides_val

    # Clean up empty arch_overrides
    if not arch_overrides or len(arch_overrides) == 0:
        entry.pop("arch_overrides", None)


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
    print(f"\nCollecting guidance updates from {len(files)} files...", flush=True)
    desired = collect_guidance_updates(files, verbose=args.verbose)

    if not desired:
        print("No guidance found in provided artifacts.")
        return 0

    # Group by config file, then test_name, creating plans on-the-fly
    by_config: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    actionable_test_count = 0

    for (test_name, arch), info in desired.items():
        plan = plan_updates_for_test(test_name, info)
        cfg = map_test_to_config_file(test_name, testing=args.testing)
        if not plan or not cfg:
            continue

        if cfg not in by_config:
            by_config[cfg] = {}
        if test_name not in by_config[cfg]:
            by_config[cfg][test_name] = {}
            actionable_test_count += 1

        by_config[cfg][test_name][arch] = plan

    if not by_config:
        print("No actionable guidance found.")
        return 0

    print(
        f"\nGenerating promotion plan for {actionable_test_count} tests...\n",
        flush=True,
    )

    # Process each config file
    for config_path in sorted(by_config.keys()):
        data = load_yaml_config(config_path)
        modified_bracket_keys: List[str] = []

        # Apply all updates for this config file
        for test_name, arch_plans in sorted(by_config[config_path].items()):
            print(
                f" - {('APPLY' if args.apply else 'PLAN ')} {os.path.basename(config_path)} :: {test_name} [archs: {', '.join(sorted(arch_plans.keys()))}] -> {arch_plans}"
            )
            bracket_key = apply_updates_to_yaml(
                data, test_name, arch_plans, args.verbose
            )
            if bracket_key:
                modified_bracket_keys.append(bracket_key)

        # Optimization pass: move common fields to top-level
        if modified_bracket_keys and not args.no_optimize:
            if args.verbose:
                print(
                    f"\nRunning optimization pass for {os.path.basename(config_path)}...\n"
                )
            test_config = data.get("test_config")
            if isinstance(test_config, dict):
                for bracket_key in modified_bracket_keys:
                    if bracket_key not in test_config:
                        continue
                    entry = test_config[bracket_key]
                    if isinstance(entry, CommentedMap):
                        all_archs = get_all_archs_for_entry(entry)
                        optimize_arch_overrides(
                            entry, all_archs, args.verbose, bracket_key
                        )

        # Write config (optimized if modifications were made)
        if args.apply:
            write_yaml_config(config_path, data)

    elapsed = time.time() - start
    print(
        f"\nFinished test promotions in {elapsed:.2f}s. Mode: {'apply' if args.apply else 'dry-run'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

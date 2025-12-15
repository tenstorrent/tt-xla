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
        plan["remove_assert_pcc_false"] = True
    elif "RAISE_PCC_099" in tags or "RAISE_PCC" in tags:
        if pcc_val_f is not None:
            adjusted_pcc = pcc_val_f - PCC_BUFFER
            new_th = min(0.99, int(adjusted_pcc * 100) / 100.0)
            if pcc_th_f is None or new_th > pcc_th_f:
                plan["set_required_pcc"] = new_th
    return plan


# Group guidance updates by config file, test name, and arch.
def group_updates_by_config(
    desired: Dict[Tuple[str, str], Dict[str, object]], testing: bool
) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, object]]]], int]:
    """
    Group guidance updates by config file, test name, and arch.
    Returns (by_config, actionable_test_count).
    """
    by_config: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    actionable_test_count = 0

    for (test_name, arch), info in desired.items():
        plan = plan_updates_for_test(test_name, info)
        cfg = map_test_to_config_file(test_name, testing=testing)
        if not plan or not cfg:
            continue

        by_config.setdefault(cfg, {})
        if test_name not in by_config[cfg]:
            by_config[cfg][test_name] = {}
            actionable_test_count += 1

        by_config[cfg][test_name][arch] = plan

    return by_config, actionable_test_count


# Get the set of all known archs for a test entry.
def get_all_archs_for_entry(entry: CommentedMap) -> set[str]:
    """Return set of all archs that exist in arch_overrides or are known defaults."""
    archs = set()
    arch_overrides = entry.get("arch_overrides") or {}
    if isinstance(arch_overrides, dict):
        archs.update(arch_overrides.keys())
    # TODO: Calculate this dynamically based on test configuration
    archs.update(["n150", "p150"])
    return archs


def copy_inline_comment_if_exists(
    source_map: CommentedMap,
    target_map: CommentedMap,
    source_key: str,
    target_key: str = None,
) -> None:
    """
    Copy inline comment from source_key to target_key using yaml_add_eol_comment.
    
    This handles ruamel.yaml's complex comment storage where inline-looking comments
    are sometimes stored as trailing comments (index 2) instead of inline (index 1).
    Uses the official API to ensure proper rendering.
    """
    if target_key is None:
        target_key = source_key

    if not (
        hasattr(source_map, "ca")
        and source_map.ca.items
        and source_key in source_map.ca.items
    ):
        return

    source_comments = source_map.ca.items[source_key]
    comment_text = None

    # Check for inline comment at index 1, otherwise extract from trailing at index 2
    if len(source_comments) > 1 and source_comments[1]:
        comment_text = str(source_comments[1]).strip()
    elif len(source_comments) > 2 and source_comments[2]:
        trailing = getattr(source_comments[2], "value", str(source_comments[2]))
        if trailing.startswith("#"):
            comment_text = trailing.split("\n")[0].strip()

    if comment_text:
        if comment_text.startswith("#"):
            comment_text = comment_text[1:].strip()
        target_map.yaml_add_eol_comment(comment_text, target_key, column=0)


def add_arch_overrides_preserving_trailing_comment(
    entry: CommentedMap, arch_overrides: CommentedMap
) -> None:
    """
    Add arch_overrides to entry while preserving trailing comments from the last key.
    
    When adding arch_overrides, any blank line + comment after the last existing key
    needs to be moved to attach after the new arch_overrides structure.
    """
    # Save trailing comment from current last key
    trailing_comment = None
    if entry and hasattr(entry, "ca") and entry.ca.items:
        last_key = list(entry.keys())[-1]
        if last_key in entry.ca.items:
            comment_list = entry.ca.items[last_key]
            if len(comment_list) > 2 and comment_list[2]:
                trailing_comment = comment_list[2]
                # Clear it from old position
                cleaned = list(comment_list)
                cleaned[2] = None
                entry.ca.items[last_key] = cleaned

    # Add arch_overrides
    entry["arch_overrides"] = arch_overrides

    # Attach trailing comment to last key in arch_overrides
    if trailing_comment and arch_overrides:
        # Find the last arch and its last key
        last_arch = list(arch_overrides.keys())[-1]
        last_arch_entry = arch_overrides[last_arch]
        if isinstance(last_arch_entry, CommentedMap) and last_arch_entry:
            last_field = list(last_arch_entry.keys())[-1]
            if last_field not in last_arch_entry.ca.items:
                last_arch_entry.ca.items[last_field] = [None, None, None, None]
            comment_list = list(last_arch_entry.ca.items.get(last_field, [None, None, None, None]))
            while len(comment_list) < 4:
                comment_list.append(None)
            # Only set trailing if no inline comment exists (to avoid conflicts)
            if not comment_list[1]:
                comment_list[2] = trailing_comment
                last_arch_entry.ca.items[last_field] = comment_list

def normalize_entry_with_arch_overrides(
    entry: CommentedMap, all_archs: set[str], fields_to_normalize: List[str]
) -> None:
    """
    Normalize entry to have arch_overrides with all archs, duplicating top-level values.
    
    Only duplicates fields when creating arch_overrides for the first time.
    If arch_overrides already exists, we leave it as-is (don't add new archs or fields).
    """
    arch_overrides_existed = "arch_overrides" in entry
    
    # If arch_overrides already exists, don't modify it during normalization
    if arch_overrides_existed:
        return
    
    # Only normalize entries that don't have arch_overrides yet
    arch_overrides = CommentedMap()

    for arch in sorted(all_archs):
        arch_overrides[arch] = CommentedMap()
        arch_entry = arch_overrides[arch]

        # Duplicate relevant fields from top-level
        for field in fields_to_normalize:
            if field in entry:
                arch_entry[field] = entry[field]
                copy_inline_comment_if_exists(entry, arch_entry, field)

    # Add arch_overrides with trailing comment preservation
    add_arch_overrides_preserving_trailing_comment(entry, arch_overrides)


def save_entry_trailing_comment(entry: CommentedMap) -> Optional[object]:
    """
    Save the trailing comment from the deepest last key in an entry.
    Recursively navigates nested structures to find it.
    """
    if not entry:
        return None
    
    def find_trailing_recursive(current_map):
        """Recursively find the trailing comment in the deepest last key."""
        if not isinstance(current_map, CommentedMap):
            return None
        
        keys = list(current_map.keys())
        if not keys:
            return None
        
        last_key = keys[-1]
        
        # Check if this level has a trailing comment
        if hasattr(current_map, "ca") and current_map.ca.items and last_key in current_map.ca.items:
            comment_list = current_map.ca.items[last_key]
            if len(comment_list) > 2 and comment_list[2]:
                # Found it at this level, but check deeper first
                pass
        
        # Check if we can go deeper
        last_value = current_map[last_key]
        if isinstance(last_value, CommentedMap) and last_value:
            # Go deeper
            deeper_comment = find_trailing_recursive(last_value)
            if deeper_comment:
                return deeper_comment
        
        # No deeper level or no comment deeper, return this level's comment
        if hasattr(current_map, "ca") and current_map.ca.items and last_key in current_map.ca.items:
            comment_list = current_map.ca.items[last_key]
            if len(comment_list) > 2 and comment_list[2]:
                return comment_list[2]
        
        return None
    
    return find_trailing_recursive(entry)


def apply_entry_trailing_comment(entry: CommentedMap, trailing_comment: object) -> None:
    """
    Apply a trailing comment to the last key in an entry.
    Navigates into nested structures (like arch_overrides) to find the true last key.
    """
    if not trailing_comment or not entry:
        return
    
    keys = list(entry.keys())
    if not keys:
        return
    
    # Find the actual last key (might be nested in arch_overrides)
    last_key = keys[-1]
    last_value = entry[last_key]
    
    target_map = entry
    target_key = last_key
    
    # If last key is arch_overrides, navigate to its last arch's last field
    if last_key == "arch_overrides" and isinstance(last_value, CommentedMap) and last_value:
        arch_keys = list(last_value.keys())
        if arch_keys:
            last_arch = arch_keys[-1]
            last_arch_entry = last_value[last_arch]
            if isinstance(last_arch_entry, CommentedMap) and last_arch_entry:
                field_keys = list(last_arch_entry.keys())
                if field_keys:
                    target_map = last_arch_entry
                    target_key = field_keys[-1]
    
    # Apply the trailing comment
    if target_key not in target_map.ca.items:
        target_map.ca.items[target_key] = [None, None, None, None]
    
    comment_list = list(target_map.ca.items.get(target_key, [None, None, None, None]))
    while len(comment_list) < 4:
        comment_list.append(None)
    
    # Only set if there's no inline comment (to avoid conflicts)
    if not comment_list[1]:
        comment_list[2] = trailing_comment
        target_map.ca.items[target_key] = comment_list


def apply_updates_to_yaml(
    data: CommentedMap,
    test_name: str,
    arch_plans: Dict[str, Dict[str, object]],
    verbose: bool,
) -> Optional[str]:
    """
    Apply changes grouped by arch, always using arch_overrides.
    Trailing comment preservation is now handled externally (after optimization).
    """
    if "test_config" not in data or data["test_config"] is None:
        return None
    test_config = data["test_config"]
    bracket_key = extract_bracket_key_from_testcase_name(test_name) or ""
    if not bracket_key or bracket_key not in test_config:
        return None
    
    entry = test_config.get(bracket_key)
    if not isinstance(entry, CommentedMap):
        entry = CommentedMap(entry or {})
        test_config[bracket_key] = entry

    if "arch_overrides" not in entry:
        return None
        
    arch_overrides = entry.get("arch_overrides")
    if not isinstance(arch_overrides, CommentedMap):
        return None

    modified = False

    for arch, plan in arch_plans.items():
        arch_entry = arch_overrides.get(arch)
        if arch_entry is None or not isinstance(arch_entry, CommentedMap):
            arch_entry = CommentedMap()
            for field in ["required_pcc", "assert_pcc"]:
                if field in entry:
                    arch_entry[field] = entry[field]
                    copy_inline_comment_if_exists(entry, arch_entry, field)
            arch_overrides[arch] = arch_entry

        if "set_required_pcc" in plan:
            new_th = float(plan["set_required_pcc"])
            old_th = arch_entry.get("required_pcc")
            if old_th is None or float(old_th) < new_th:
                if verbose:
                    print(f"   - Setting arch_overrides.{arch}.required_pcc: {old_th} -> {new_th} for {bracket_key}")
                arch_entry["required_pcc"] = new_th
                modified = True

        if plan.get("remove_assert_pcc_false"):
            if arch_entry.get("assert_pcc") is False:
                if verbose:
                    print(f"   - Removing arch_overrides.{arch}.assert_pcc:false for {bracket_key}")
                arch_entry.pop("assert_pcc", None)
                modified = True
                if not arch_entry:
                    arch_overrides.pop(arch, None)

    # Clean up empty arch_overrides
    if not arch_overrides or len(arch_overrides) == 0:
        entry.pop("arch_overrides", None)

    return bracket_key if modified else None


# Save trailing comments from all test entries before modifications.
def save_trailing_comments_for_tests(
    test_config: Dict, test_names: List[str]
) -> Dict[str, object]:
    """Save trailing comments from entries that will be modified."""
    saved: Dict[str, object] = {}
    for test_name in test_names:
        bracket_key = extract_bracket_key_from_testcase_name(test_name)
        if bracket_key and bracket_key in test_config:
            entry = test_config[bracket_key]
            if isinstance(entry, CommentedMap):
                trailing = save_entry_trailing_comment(entry)
                if trailing:
                    saved[bracket_key] = trailing
    return saved


# Normalize test entries by duplicating top-level fields to arch_overrides.
def normalize_tests_for_modification(
    test_config: Dict, test_names: List[str], verbose: bool
) -> None:
    """Normalize test entries to have arch_overrides with duplicated fields."""
    for test_name in test_names:
        bracket_key = extract_bracket_key_from_testcase_name(test_name)
        if bracket_key and bracket_key in test_config:
            entry = test_config[bracket_key]
            if isinstance(entry, CommentedMap):
                all_archs = get_all_archs_for_entry(entry)
                normalize_entry_with_arch_overrides(
                    entry, all_archs, ["required_pcc", "assert_pcc"]
                )


# Apply all updates for a config file and return list of modified bracket keys.
def apply_all_updates(
    data: CommentedMap,
    test_plans: Dict[str, Dict[str, Dict[str, object]]],
    apply_mode: bool,
    verbose: bool,
    config_path: str,
) -> List[str]:
    """Apply all updates for a config file."""
    modified_bracket_keys: List[str] = []
    for test_name, arch_plans in sorted(test_plans.items()):
        mode_str = "APPLY" if apply_mode else "PLAN "
        archs_str = ", ".join(sorted(arch_plans.keys()))
        print(
            f" - {mode_str} {os.path.basename(config_path)} :: {test_name} "
            f"[archs: {archs_str}] -> {arch_plans}"
        )
        bracket_key = apply_updates_to_yaml(data, test_name, arch_plans, verbose)
        if bracket_key:
            modified_bracket_keys.append(bracket_key)
    return modified_bracket_keys


# Run optimization pass on all modified tests.
def optimize_modified_tests(
    test_config: Dict, modified_bracket_keys: List[str], verbose: bool, config_path: str
) -> None:
    """Run optimization pass to consolidate common fields."""
    if verbose:
        print(f"\nRunning optimization pass for {os.path.basename(config_path)}...\n")
    for bracket_key in modified_bracket_keys:
        if bracket_key not in test_config:
            continue
        entry = test_config[bracket_key]
        if isinstance(entry, CommentedMap):
            all_archs = get_all_archs_for_entry(entry)
            optimize_arch_overrides(entry, all_archs, verbose, bracket_key)


# Restore trailing comments to modified tests after optimization.
def restore_trailing_comments_for_tests(
    test_config: Dict, modified_bracket_keys: List[str], saved_comments: Dict[str, object]
) -> None:
    """Restore trailing comments after optimization."""
    for bracket_key in modified_bracket_keys:
        if bracket_key in saved_comments and bracket_key in test_config:
            entry = test_config[bracket_key]
            if isinstance(entry, CommentedMap):
                apply_entry_trailing_comment(entry, saved_comments[bracket_key])


def optimize_arch_overrides(
    entry: CommentedMap, all_archs: set[str], verbose: bool, bracket_key: str
) -> None:
    """
    Move fields to top-level if common across all archs, preserving comments.
    """
    arch_overrides = entry.get("arch_overrides")
    if not arch_overrides or not isinstance(arch_overrides, dict):
        return

    all_fields: set[str] = set()
    for arch_entry in arch_overrides.values():
        if isinstance(arch_entry, dict):
            all_fields.update(arch_entry.keys())

    fields_to_move: Dict[str, object] = {}
    for field in all_fields:
        field_values: Dict[str, object] = {}
        
        for arch in arch_overrides.keys():
            arch_entry = arch_overrides.get(arch)
            if isinstance(arch_entry, dict) and field in arch_entry:
                field_values[arch] = arch_entry[field]

        if len(field_values) == len(all_archs):
            values = list(field_values.values())
            first_value = values[0]
            if all(v == first_value for v in values):
                if field not in entry or entry[field] != first_value:
                    fields_to_move[field] = first_value

    if fields_to_move:
        for field, common_value in fields_to_move.items():
            if verbose:
                print(f" - Optimizing: moving {field}={common_value} to top-level for {bracket_key}")
            
            # Move with comment preservation
            # Get comment from last arch entry (in YAML order)
            source_arch = list(arch_overrides.keys())[-1]
            source_entry = arch_overrides[source_arch]
            if isinstance(source_entry, CommentedMap) and field in source_entry:
                entry[field] = common_value
                copy_inline_comment_if_exists(source_entry, entry, field)
            else:
                entry[field] = common_value

            # Remove from all arch_overrides
            for arch in all_archs:
                arch_entry = arch_overrides.get(arch)
                if isinstance(arch_entry, dict) and field in arch_entry:
                    arch_entry.pop(field, None)
                    if not arch_entry:
                        arch_overrides.pop(arch, None)

        # Reorder: move arch_overrides to end
        if "arch_overrides" in entry and entry["arch_overrides"]:
            arch_overrides_val = entry.pop("arch_overrides")
            entry["arch_overrides"] = arch_overrides_val

    # Clean up empty arch_overrides
    if not arch_overrides or len(arch_overrides) == 0:
        entry.pop("arch_overrides", None)


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

    by_config, actionable_test_count = group_updates_by_config(desired, args.testing)

    if not by_config:
        print("No actionable guidance found.")
        return 0

    print(f"\nGenerating promotion plan for {actionable_test_count} tests...\n", flush=True)

    for config_path in sorted(by_config.keys()):
        data = load_yaml_config(config_path)
        test_config = data.get("test_config")
        if not isinstance(test_config, dict):
            continue

        # Get list of test names to modify
        test_names = list(by_config[config_path].keys())
        
        # STEP 1: Save trailing comments before modifications
        saved_trailing_comments = save_trailing_comments_for_tests(test_config, test_names)
        
        # STEP 2: Normalize entries (duplicate fields to arch_overrides)
        normalize_tests_for_modification(test_config, test_names, args.verbose)
        
        # STEP 3: Apply updates
        modified_bracket_keys = apply_all_updates(
            data, by_config[config_path], args.apply, args.verbose, config_path
        )
        
        # STEP 4: Optimization pass
        if modified_bracket_keys and not args.no_optimize:
            optimize_modified_tests(test_config, modified_bracket_keys, args.verbose, config_path)
        
        # STEP 5: Restore trailing comments after optimization
        restore_trailing_comments_for_tests(test_config, modified_bracket_keys, saved_trailing_comments)

        # Write changes
        if args.apply:
            write_yaml_config(config_path, data)

    elapsed = time.time() - start
    print(f"\nFinished test promotions in {elapsed:.2f}s. Mode: {'apply' if args.apply else 'dry-run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
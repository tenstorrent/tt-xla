# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import csv
import glob
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional

# Small buffer to avoid enabling or tightening thresholds when near boundary.
PCC_BUFFER = 0.004


# Find and normalize XML paths/patterns into a sorted, deduplicated list of files.
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
    # Deduplicate and sort for stable output
    uniq = sorted({os.path.abspath(p) for p in files})
    return [p for p in uniq if os.path.isfile(p)]


# Safely parse a Python-literal dict embedded as a string in XML property value.
def parse_tags_value(value: str) -> Dict:
    # Properties appear as a Python dict in a string (single quotes, True/False, None)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


# Convert any value to string; map None to empty string.
def to_str(value) -> str:
    if value is None:
        return ""
    return str(value)


# Convert values to printable cells; map None/empty string to 'N/A'.
def to_cell(value) -> str:
    # Convert None or empty string to a visible placeholder
    if value is None:
        return "N/A"
    s = str(value)
    return "N/A" if s == "" else s


# Convert a value to a printable cell with column-aware formatting.
def format_cell(column: str, value) -> str:
    if column == "pcc_assertion_enabled":
        # Normalize booleans or boolean-like strings to PCC_EN/PCC_DIS using shared helper
        parsed = _to_bool(value)
        if parsed is True:
            return "PCC_EN"
        if parsed is False:
            return "PCC_DIS"
        return to_cell(value)
    return to_cell(value)


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    sval = str(value).strip().lower()
    if sval in ("true", "1", "yes"):
        return True
    if sval in ("false", "0", "no"):
        return False
    return None


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


# Build derived tags string using PCC-related heuristics. Eventually these might move to test infra
# but for now, keep it here.
def compute_row_tags(rec: Dict) -> str:
    tags: List[str] = []
    pcc = _to_float(rec.get("pcc"))
    pcc_th = _to_float(rec.get("pcc_threshold"))
    pcc_en = _to_bool(rec.get("pcc_assertion_enabled"))

    if None not in (pcc, pcc_th, pcc_en):
        # If PCC is safely above threshold+buffer but disabled, suggest enabling
        if (pcc > (pcc_th + PCC_BUFFER)) and (pcc_en is False):
            if pcc_th >= 0.99:
                tags.append("ENABLE_PCC_099")
            else:
                tags.append("ENABLE_PCC")

        # If PCC clears the next 0.01 step above current threshold by a buffer, flag that it
        # should be raised. Only flag RAISE_PCC when current threshold is below 0.99.
        if pcc_th < 0.99:
            # Compute the next centesimal threshold (e.g., 0.98 -> 0.99), capped at 0.99
            next_level = min(0.99, (math.floor(pcc_th * 100) + 1) / 100.0)
            if pcc > (next_level + PCC_BUFFER):
                if next_level >= 0.99:
                    tags.append("RAISE_PCC_099")
                else:
                    tags.append("RAISE_PCC")

    return ",".join(tags) if tags else ""


# Build and return the CLI argument parser.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize selected fields from JUnit XML 'tags' properties."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--xml",
        nargs="+",
        help="One or more XML files, directories, or globs (e.g. ./*/*.xml). Alternative to --run-id; cannot be combined.",
    )
    group.add_argument(
        "--run-id",
        help="GitHub run-id to download report artifacts, then summarize them. Alternative to --xml; cannot be combined.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print header row",
    )
    parser.add_argument(
        "--group",
        help="Filter rows where group equals this value",
    )
    parser.add_argument(
        "--arch",
        help="Filter rows where arch equals this value",
    )
    parser.add_argument(
        "--parallelism",
        help="Filter rows where parallelism equals this value",
    )
    parser.add_argument(
        "--runmode",
        help="Filter rows where run_mode equals this value (e.g., inference or training)",
    )
    parser.add_argument(
        "--model-type",
        help="Filter rows where model_type equals this value (e.g., vision, language, other)",
    )
    parser.add_argument(
        "--bringup-status",
        dest="bringup_status",
        help="Filter rows where bringup_status equals this value (e.g., incorrect_result, passed)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV (comma-separated). Ignores alignment.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication; show all occurrences across XMLs (useful for debug).",
    )
    return parser


# Resolve input patterns from --xml or --run-id, downloading artifacts if needed.
def resolve_input_patterns(
    xml_patterns: Optional[List[str]], run_id: Optional[str]
) -> List[str]:
    patterns: List[str] = []
    if xml_patterns:
        patterns.extend(xml_patterns)
        return patterns
    if run_id:
        out_dir = f"run_id_{run_id}_reports"
        # If directory missing or has no XMLs, fetch artifacts first
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


# Define the output columns and their order.
def get_columns() -> List[str]:
    return [
        "specific_test_case",
        "model_type",
        "group",
        "arch",
        "bringup_status",
        "pcc",
        "pcc_threshold",
        "pcc_assertion_enabled",
        "parallelism",
        "time",
        "tags",
    ]


# Define shorter header labels for the columns.
def get_header_labels() -> List[str]:
    keys = get_columns()
    label_map = {
        "pcc_threshold": "pcc_thres",
        "pcc_assertion_enabled": "pcc_en",
    }
    return [label_map.get(k, k) for k in keys]


# Iterate testcases and build a flat record per testcase by merging tags and group.
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
        record["time"] = tc.get("time")

        # Derive model_type from model_info.task
        model_info = record.get("model_info") or {}
        task_value = model_info.get("task")
        record["model_type"] = compute_model_type(task_value)
        yield record


# Check whether a record matches optional CLI filters (case-insensitive exact match).
def record_matches_filters(
    record: Dict,
    group_filter: Optional[str],
    arch_filter: Optional[str],
    parallelism_filter: Optional[str],
    runmode_filter: Optional[str],
    model_type_filter: Optional[str],
    bringup_status_filter: Optional[str],
) -> bool:
    if group_filter is not None:
        if to_str(record.get("group")).lower() != group_filter.lower():
            return False
    if arch_filter is not None:
        if to_str(record.get("arch")).lower() != arch_filter.lower():
            return False
    if parallelism_filter is not None:
        if to_str(record.get("parallelism")).lower() != parallelism_filter.lower():
            return False
    if runmode_filter is not None:
        if to_str(record.get("run_mode")).lower() != runmode_filter.lower():
            return False
    if model_type_filter is not None:
        if to_str(record.get("model_type")).lower() != model_type_filter.lower():
            return False
    if bringup_status_filter is not None:
        if (
            to_str(record.get("bringup_status")).lower()
            != bringup_status_filter.lower()
        ):
            return False
    return True


# Map a model task (e.g., 'cv_image_cls', 'nlp_causal_lm') to a high-level model type.
# NOTE: This mapping is heuristic and subject to change; may be refactored later.
def compute_model_type(task_value: Optional[str]) -> str:
    if task_value is None:
        return "unknown"
    task_str = str(task_value)
    task_lc = task_str.lower()
    if (
        task_lc.startswith("cv_")
        or task_lc.startswith("mm_")
        or "_map_" in task_lc
        or task_lc == "conditional_generation"
    ):
        return "vision"
    if task_lc.startswith("nlp_") or task_lc.startswith("audio_"):
        return "language"
    if task_lc == "atomic_ml":
        return "other"
    return "unknown"


# Extract and convert testsuite timestamp to epoch seconds, if present.
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


# Parse all files, filter records (by test prefix and optional filters), and produce rows.
def collect_rows_from_files(
    files: List[str],
    columns: List[str],
    group_filter: Optional[str],
    arch_filter: Optional[str],
    parallelism_filter: Optional[str],
    runmode_filter: Optional[str],
    model_type_filter: Optional[str],
    bringup_status_filter: Optional[str],
    dedupe_enabled: bool = True,
) -> List[List[str]]:
    # Collect rows once; optionally dedupe by testsuite timestamp (latest wins).
    latest_by_key: Dict[tuple, List[str]] = {}
    score_by_key: Dict[tuple, float] = {}
    rows: List[List[str]] = []
    arch_idx = columns.index("arch")

    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            print(f"Failed to parse XML: {path}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error reading file: {path}: {e}", file=sys.stderr)
            continue

        # Use testsuite timestamp for deduplication; missing timestamps default to 0.0
        score = get_suite_timestamp(tree) or 0.0

        for rec in iter_test_records(tree):
            test_name = to_str(rec.get("specific_test_case", ""))

            # We only care about model tests via test_models.py for now.
            if not test_name.startswith("test_all_models"):
                continue

            # Apply optional CLI filters
            if not record_matches_filters(
                rec,
                group_filter,
                arch_filter,
                parallelism_filter,
                runmode_filter,
                model_type_filter,
                bringup_status_filter,
            ):
                continue

            # Compute derived tags prior to rendering
            rec["tags"] = compute_row_tags(rec)
            row = [format_cell(c, rec.get(c)) for c in columns]
            safe_row = [
                c.replace("\n", " ").replace("\r", " ").replace("\t", " ") for c in row
            ]

            if dedupe_enabled:
                arch_val = to_str(rec.get("arch", ""))
                key = (test_name, arch_val)
                prev_score = score_by_key.get(key)
                if prev_score is None or score >= prev_score:
                    score_by_key[key] = score
                    latest_by_key[key] = safe_row
            else:
                rows.append(safe_row)

    if dedupe_enabled:
        # Deterministic output: sort by (test name, arch)
        return [latest_by_key[k] for k in sorted(latest_by_key.keys())]
    # Deterministic output without dedupe: sort by (specific_test_case, arch)
    return sorted(rows, key=lambda r: (r[0], r[arch_idx]))


# Compute maximum widths for each column based on header and rows.
def compute_column_widths(columns: List[str], rows: List[List[str]]) -> List[int]:
    col_widths = [len(col) for col in columns]
    for r in rows:
        for idx, cell in enumerate(r):
            if len(cell) > col_widths[idx]:
                col_widths[idx] = len(cell)
    return col_widths


# Print the header (unless suppressed) and rows with per-column padding and separator.
def print_table(
    columns: List[str], rows: List[List[str]], sep: str, no_header: bool, csv_mode: bool
) -> None:
    if csv_mode:
        writer = csv.writer(sys.stdout, lineterminator="\n")
        if not no_header:
            writer.writerow(columns)
        for r in rows:
            writer.writerow(r)
        return
    col_widths = compute_column_widths(columns, rows)
    if not no_header:
        header_cells = [columns[i].ljust(col_widths[i]) for i in range(len(columns))]
        print(sep.join(header_cells))
    for r in rows:
        out_cells = [r[i].ljust(col_widths[i]) for i in range(len(columns))]
        print(sep.join(out_cells))


# Program entry: parse args, collect files and rows, and print the formatted table.
def main(argv: Iterable[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv))

    input_patterns = resolve_input_patterns(args.xml, args.run_id)
    if not input_patterns:
        print("No XML sources resolved. Use --xml or --run-id.", file=sys.stderr)
        return 2
    files = collect_input_files(input_patterns)
    if not files:
        print("No XML files matched.", file=sys.stderr)
        return 1

    columns = get_columns()
    header_labels = get_header_labels()
    rows = collect_rows_from_files(
        files,
        columns,
        group_filter=args.group,
        arch_filter=args.arch,
        parallelism_filter=args.parallelism,
        runmode_filter=args.runmode,
        model_type_filter=args.model_type,
        bringup_status_filter=args.bringup_status,
        dedupe_enabled=not args.no_dedupe,
    )
    print_table(
        header_labels, rows, sep="  ", no_header=args.no_header, csv_mode=args.csv
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

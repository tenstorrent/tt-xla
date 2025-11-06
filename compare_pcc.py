#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

PCC_REGEX = re.compile(r"pcc\s*=\s*([0-9]*\.?[0-9]+)")


def expand_xml_paths(path_or_glob: str) -> List[str]:
    # Support either a directory or a glob like /path/*.xml
    if os.path.isdir(path_or_glob):
        pattern = os.path.join(path_or_glob, "**", "*.xml")
        return sorted(glob.glob(pattern, recursive=True))
    else:
        return sorted(glob.glob(path_or_glob, recursive=True))


def try_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def extract_pcc_from_properties(properties_elem: ET.Element) -> Optional[float]:
    # Prefer pcc inside the 'tags' property (stringified Python dict)
    for prop in properties_elem.findall("property"):
        name = prop.get("name")
        value = prop.get("value", "")

        if name == "tags" and value:
            try:
                tags = ast.literal_eval(value)
                if isinstance(tags, dict) and "pcc" in tags:
                    pcc_val = tags.get("pcc")
                    if isinstance(pcc_val, (int, float)):
                        return float(pcc_val)
            except Exception:
                pass  # fall through to other methods

    # Look for explicit pcc property
    for prop in properties_elem.findall("property"):
        if prop.get("name") == "pcc":
            pcc = try_float(prop.get("value", ""))
            if pcc is not None:
                return pcc

    # Parse pcc from error_message, e.g. "Calculated: pcc=0.872334..."
    for prop in properties_elem.findall("property"):
        if prop.get("name") == "error_message":
            match = PCC_REGEX.search(prop.get("value", ""))
            if match:
                return try_float(match.group(1))

    return None


def extract_pcc_from_testcase(testcase: ET.Element) -> Optional[float]:
    # Properties under the testcase element
    properties = testcase.find("properties")
    if properties is not None:
        pcc = extract_pcc_from_properties(properties)
        if pcc is not None:
            return pcc

    # Sometimes properties might be at the testsuite level; skip for simplicity
    return None


def parse_junit_file(xml_path: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Warning: failed to parse '{xml_path}': {e}", file=sys.stderr)
        return result

    for testcase in root.findall(".//testcase"):
        name = testcase.get("name") or ""
        if not name:
            continue

        pcc = extract_pcc_from_testcase(testcase)
        if pcc is not None:
            # If duplicates occur, last one wins
            result[name] = pcc

    return result


def parse_junit_inputs(paths_or_globs: List[str]) -> Dict[str, float]:
    all_results: Dict[str, float] = {}
    all_files: List[str] = []
    for p in paths_or_globs:
        all_files.extend(expand_xml_paths(p))
    for f in sorted(set(all_files)):
        file_results = parse_junit_file(f)
        all_results.update(file_results)
    return all_results


def format_float(val: Optional[float]) -> str:
    return "-" if val is None else f"{val:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCCs from pytest JUnit XMLs before vs after."
    )
    parser.add_argument(
        "--before",
        required=True,
        nargs="+",
        help=(
            "One or more BEFORE directories/globs. Examples: /path/dir  /path/dir/*.xml"
        ),
    )
    parser.add_argument(
        "--after",
        required=True,
        nargs="+",
        help=(
            "One or more AFTER directories/globs. Examples: /path/dir2  /path/dir2/*.xml"
        ),
    )
    parser.add_argument(
        "--sort",
        default="change",
        choices=["change", "name", "before", "after", "improved"],
        help="Sort output by column. 'improved' sorts by change desc, then after desc.",
    )
    parser.add_argument(
        "--only-improved",
        action="store_true",
        help="Show only tests where pcc_after > pcc_before.",
    )
    args = parser.parse_args()

    before_map = parse_junit_inputs(args.before)
    after_map = parse_junit_inputs(args.after)

    all_names = sorted(set(before_map.keys()).union(after_map.keys()))
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []

    for name in all_names:
        b = before_map.get(name)
        a = after_map.get(name)
        change = (a - b) if (a is not None and b is not None) else None
        rows.append((name, b, a, change))

    if args.only_improved:
        rows = [
            r for r in rows if r[1] is not None and r[2] is not None and r[2] > r[1]
        ]

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    elif args.sort == "before":
        rows.sort(key=lambda r: (-1 if r[1] is None else r[1]))
    elif args.sort == "after":
        rows.sort(key=lambda r: (-1 if r[2] is None else r[2]))
    elif args.sort == "change":
        # Place Nones at the end, then descending by change
        rows.sort(key=lambda r: (r[3] is None, -(r[3] or -1e18)))
    elif args.sort == "improved":
        # Improved first (change>0), then by change desc, then after desc
        rows.sort(
            key=lambda r: (
                not (r[3] is not None and r[3] > 0),
                -(r[3] or -1e18),
                -(r[2] or -1e18),
            )
        )

    # Compute column widths
    name_width = (
        max([len("test_name")] + [len(r[0]) for r in rows])
        if rows
        else len("test_name")
    )
    headers = ("test_name", "pcc_before", "pcc_after", "pcc_change")

    print(
        f"{headers[0]:<{name_width}}  {headers[1]:>12}  {headers[2]:>12}  {headers[3]:>12}"
    )
    print(f"{'-'*name_width}  {'-'*12}  {'-'*12}  {'-'*12}")

    for name, b, a, c in rows:
        print(
            f"{name:<{name_width}}  {format_float(b):>12}  {format_float(a):>12}  {format_float(c):>12}"
        )


if __name__ == "__main__":
    main()

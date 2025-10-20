#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import difflib
import json
import os
import sys
from typing import Dict, List, Set, Tuple


def list_json_files(directory: str) -> Set[str]:
    try:
        entries = os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return set()
    return {
        name
        for name in entries
        if name.endswith(".json") and os.path.isfile(os.path.join(directory, name))
    }


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def dump_pretty(obj) -> List[str]:
    return json.dumps(obj, indent=2, sort_keys=True).splitlines(keepends=False)


def compare_dirs(dir_a: str, dir_b: str, show_diff: bool) -> int:
    files_a = list_json_files(dir_a)
    files_b = list_json_files(dir_b)

    missing_in_b = sorted(files_a - files_b)
    missing_in_a = sorted(files_b - files_a)

    status = 0

    if missing_in_b:
        status = 1
        print(f"Only in {dir_a}:")
        for name in missing_in_b:
            print(f"  {name}")

    if missing_in_a:
        status = 1
        print(f"Only in {dir_b}:")
        for name in missing_in_a:
            print(f"  {name}")

    common = sorted(files_a & files_b)
    mismatches: List[Tuple[str, str]] = []

    for name in common:
        path_a = os.path.join(dir_a, name)
        path_b = os.path.join(dir_b, name)
        try:
            obj_a = read_json(path_a)
            obj_b = read_json(path_b)
        except Exception as e:
            status = 1
            print(f"Error reading JSON for {name}: {e}")
            continue

        if obj_a != obj_b:
            status = 1
            mismatches.append((path_a, path_b))

    if mismatches:
        print("Content mismatches detected:")
        for path_a, path_b in mismatches:
            print(f"- {os.path.basename(path_a)}")
            if show_diff:
                try:
                    a_lines = dump_pretty(read_json(path_a))
                    b_lines = dump_pretty(read_json(path_b))
                    udiff = difflib.unified_diff(
                        a_lines,
                        b_lines,
                        fromfile=path_a,
                        tofile=path_b,
                        lineterm="",
                    )
                    for line in udiff:
                        print(line)
                except Exception as e:
                    print(f"  (failed to diff: {e})")

    if status == 0:
        print(f"OK: {len(common)} common files, no differences found.")
    else:
        print(
            f"FAIL: {len(common)} common files compared, {len(missing_in_a)} only-in-B, {len(missing_in_b)} only-in-A, {len(mismatches)} mismatches."
        )

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Compare two directories of test metadata JSON files."
    )
    parser.add_argument("dir_a", help="Path to first directory (e.g., before)")
    parser.add_argument("dir_b", help="Path to second directory (e.g., after)")
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show unified diffs for files with mismatched content",
    )
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    rc = compare_dirs(dir_a, dir_b, show_diff=args.diff)
    sys.exit(rc)


if __name__ == "__main__":
    main()

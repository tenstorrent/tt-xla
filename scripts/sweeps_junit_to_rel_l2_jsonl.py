#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Convert a tt-forge-sweeps JUnit XML (with `tags` property containing
`relative_l2` and `pcc`) into the JSONL format consumed by
scripts/compare_rel_l2.py and scripts/report_rel_l2.py.

Each emitted line: {"test_id": ..., "rel_l2": ..., "pcc": ...}

Usage:
    python3 scripts/sweeps_junit_to_rel_l2_jsonl.py before.xml > before.jsonl
    python3 scripts/sweeps_junit_to_rel_l2_jsonl.py after.xml  > after.jsonl
    python3 scripts/compare_rel_l2.py before.jsonl after.jsonl
"""

import argparse
import ast
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def extract(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # <testsuites><testsuite><testcase name=...><properties><property name="tags" value="{...}"/>
    for tc in root.iter("testcase"):
        test_id = tc.get("name", "")
        tags_val = None
        for prop in tc.iter("property"):
            if prop.get("name") == "tags":
                tags_val = prop.get("value")
                break
        if not tags_val:
            continue
        try:
            tags = ast.literal_eval(tags_val)
        except (ValueError, SyntaxError) as exc:
            print(f"WARN: skip {test_id}: tags parse failed: {exc}", file=sys.stderr)
            continue
        rel_l2 = tags.get("relative_l2")
        pcc = tags.get("pcc")
        if rel_l2 is None or pcc is None:
            continue
        yield {"test_id": test_id, "rel_l2": float(rel_l2), "pcc": float(pcc)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xml", type=Path, help="sweeps JUnit XML")
    args = p.parse_args()

    count = 0
    for entry in extract(args.xml):
        print(json.dumps(entry))
        count += 1
    print(f"# wrote {count} entries", file=sys.stderr)


if __name__ == "__main__":
    main()

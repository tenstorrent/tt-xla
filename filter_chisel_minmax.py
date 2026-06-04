#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Filter a full chisel report JSONL down to min/max PCC per isolated failure.

Accepts an entire chisel report: it first keeps only lines that contain both
``numerics_fail`` and ``isolated`` (the substring equivalent of
``grep numerics_fail | grep isolated``), then collapses each op to its min/max.

Each kept line is one chisel numerics record for a single device, e.g.:

    {"op":"ttnn.softmax","check":"numerics","ssa":"%200","binary_id":1,
     "program_name":"main","program_index":0,
     "payload":{"status":"...","mode":"isolated","pcc":0.258,...,"device_id":0}}

All devices that ran the same op share the identity
(op, check, ssa, binary_id, program_name, program_index) and differ only in
their payload (pcc, device_id, ...). For each such op this writes the original
JSON lines, verbatim:

  - two lines (lowest then highest PCC) when the op has a distinct min and max,
  - one line when there is no spread to report: either every record's pcc is
    null, or the lowest and highest pcc are equal (e.g. a single device).

Usage:
    python3 filter_chisel_minmax.py [INPUT.jsonl] [OUTPUT.log]
"""
import argparse
import json
import sys

# Fields that identify a single op instance (everything outside `payload`).
KEY_FIELDS = ("op", "check", "ssa", "binary_id", "program_name", "program_index")

# Substring pre-filter, applied to the raw line (== `grep A | grep B`).
REQUIRED_SUBSTRINGS = ("numerics_fail", "isolated")

DEFAULT_INPUT = "deepseek_v3_1_benchmark_report_failures_isolated.jsonl"
DEFAULT_OUTPUT = "deepseek_v3_1_benchmark_report_failures_minmax.log"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Preserve first-seen order of ops so the output is stable/diffable.
    order = []
    groups = {}  # key -> {"first": raw, "lo": (pcc, raw)|None, "hi": (pcc, raw)|None}

    skipped_parse = 0
    kept = 0

    with open(args.input, "r") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            # Pre-filter: grep numerics_fail | grep isolated
            if not all(s in raw for s in REQUIRED_SUBSTRINGS):
                continue
            kept += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                skipped_parse += 1
                print(
                    f"[warn] line {lineno}: not valid JSON, skipping", file=sys.stderr
                )
                continue

            key = tuple(rec.get(k) for k in KEY_FIELDS)
            grp = groups.get(key)
            if grp is None:
                grp = {"first": raw, "lo": None, "hi": None}
                groups[key] = grp
                order.append(key)

            pcc = rec.get("payload", {}).get("pcc")
            if isinstance(pcc, (int, float)):
                if grp["lo"] is None or pcc < grp["lo"][0]:
                    grp["lo"] = (pcc, raw)
                if grp["hi"] is None or pcc > grp["hi"][0]:
                    grp["hi"] = (pcc, raw)

    two_line_ops = 0
    one_line_ops = 0
    with open(args.output, "w") as out:
        for key in order:
            grp = groups[key]
            if grp["lo"] is None:
                # No numeric pcc anywhere in this op (all null/missing): 1 line.
                out.write(grp["first"] + "\n")
                one_line_ops += 1
            elif grp["lo"][0] == grp["hi"][0]:
                # No spread (single device or all-equal pcc): 1 line.
                out.write(grp["lo"][1] + "\n")
                one_line_ops += 1
            else:
                out.write(grp["lo"][1] + "\n")
                out.write(grp["hi"][1] + "\n")
                two_line_ops += 1

    total_lines = two_line_ops * 2 + one_line_ops
    print(
        f"[done] {kept} matching record(s), {len(order)} op(s) -> {total_lines} "
        f"lines written to {args.output} "
        f"({two_line_ops} with min/max, {one_line_ops} single-line)"
    )
    if skipped_parse:
        print(f"[done] skipped {skipped_parse} unparseable line(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

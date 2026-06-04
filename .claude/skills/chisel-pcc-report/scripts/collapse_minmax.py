#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Collapse a raw chisel report JSONL into per-op min/max PCC failures.

This is the deterministic core of the ``chisel-pcc-report`` skill. It does the
heavy lifting on large reports (100k+ lines) so the model never has to parse the
raw JSONL by hand. It produces two things:

  1. ``--filtered-out`` : a verbatim min/max JSONL (same shape as
     ``filter_chisel_minmax.py`` output) for future reference -- two lines per op
     (lowest then highest PCC), or one line when there is no spread.
  2. ``--summary-out`` (or stdout) : a compact JSON object the skill consumes to
     build the human-readable report. Failures are pre-sorted worst-first.

Identity of one op instance is the tuple
``(op, check, ssa, binary_id, program_name, program_index)``; the same op recurs
once per ``device_id`` and differs only in its payload (pcc, atol, device_id).

The summary JSON has the shape::

    {
      "input": "<path>",
      "totals": {
        "isolated_numerics_failures": <int>,   # unique failing op instances
        "accumulated_numerics_failures": <int>, # context only, not detailed
        "harness_issues": <int>
      },
      "failures": [                             # sorted by min pcc ascending
        {
          "op": "ttnn.softmax", "ssa": "%200", "check": "numerics",
          "binary_id": 1, "program_name": "main", "program_index": 0,
          "num_devices": 32,
          "min": {"pcc": 0.005, "atol": 0.99, "rtol": null, "device_id": 17},
          "max": {"pcc": 0.30,  "atol": 0.98, "rtol": null, "device_id": 6}
        }, ...
      ],
      "harness_issues": [                       # shape/dtype/chisel_bug etc.
        {"op": "ttnn.to_device", "check": "mlir_vs_runtime_tensor",
         "status": "shape_mismatch", "count": 4,
         "example": { <full payload of one record> }}, ...
      ],
      "binary_ids_with_failures": [1, 3]        # graphs should be sought per id
    }

Usage:
    python3 collapse_minmax.py REPORT.jsonl \
        --filtered-out pcc_filtered_<tag>.jsonl \
        --summary-out  pcc_summary_<tag>.json
"""
import argparse
import json
import sys

# Fields that identify a single op instance (everything outside `payload`).
KEY_FIELDS = ("op", "check", "ssa", "binary_id", "program_name", "program_index")

# Payload statuses that signal a harness/compile problem (not numerics drift).
HARNESS_STATUSES = {"shape_mismatch", "dtype_mismatch", "chisel_bug", "error"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collapse a raw chisel report to per-op min/max PCC failures."
    )
    parser.add_argument("input", help="raw chisel report JSONL")
    parser.add_argument(
        "--filtered-out",
        default=None,
        help="path to write the verbatim min/max JSONL (optional)",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="path to write the summary JSON (default: stdout)",
    )
    args = parser.parse_args()

    order = []  # first-seen identity keys, for stable output
    groups = {}  # key -> aggregation dict
    harness = {}  # (op, check, status) -> {"count", "example"}
    accumulated_fail = 0
    skipped_parse = 0

    with open(args.input, "r") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                skipped_parse += 1
                print(
                    f"[warn] line {lineno}: not valid JSON, skipping", file=sys.stderr
                )
                continue

            payload = rec.get("payload") or {}
            status = payload.get("status")

            # --- harness / compile issues -------------------------------------
            if status in HARNESS_STATUSES:
                hkey = (rec.get("op"), rec.get("check"), status)
                h = harness.get(hkey)
                if h is None:
                    harness[hkey] = {"count": 1, "example": rec}
                else:
                    h["count"] += 1
                continue

            # --- numerics failures --------------------------------------------
            if status != "numerics_fail":
                continue
            if payload.get("mode") != "isolated":
                if payload.get("mode") == "accumulated":
                    accumulated_fail += 1
                continue

            key = tuple(rec.get(k) for k in KEY_FIELDS)
            grp = groups.get(key)
            if grp is None:
                grp = {
                    "op": rec.get("op"),
                    "check": rec.get("check"),
                    "ssa": rec.get("ssa"),
                    "binary_id": rec.get("binary_id"),
                    "program_name": rec.get("program_name"),
                    "program_index": rec.get("program_index"),
                    "num_devices": 0,
                    "lo": None,  # (pcc, payload, raw)
                    "hi": None,
                    "first_raw": raw,
                }
                groups[key] = grp
                order.append(key)

            grp["num_devices"] += 1
            pcc = payload.get("pcc")
            if isinstance(pcc, (int, float)):
                if grp["lo"] is None or pcc < grp["lo"][0]:
                    grp["lo"] = (pcc, payload, raw)
                if grp["hi"] is None or pcc > grp["hi"][0]:
                    grp["hi"] = (pcc, payload, raw)

    # --- write the verbatim filtered JSONL (min then max per op) --------------
    if args.filtered_out:
        with open(args.filtered_out, "w") as out:
            for key in order:
                grp = groups[key]
                if grp["lo"] is None:
                    out.write(grp["first_raw"] + "\n")
                elif grp["lo"][0] == grp["hi"][0]:
                    out.write(grp["lo"][2] + "\n")
                else:
                    out.write(grp["lo"][2] + "\n")
                    out.write(grp["hi"][2] + "\n")

    # --- build the summary ----------------------------------------------------
    def side(entry):
        if entry is None:
            return None
        pcc, payload, _ = entry
        return {
            "pcc": pcc,
            "atol": payload.get("atol"),
            "rtol": payload.get("rtol"),
            "device_id": payload.get("device_id"),
        }

    failures = []
    for key in order:
        grp = groups[key]
        failures.append(
            {
                "op": grp["op"],
                "ssa": grp["ssa"],
                "check": grp["check"],
                "binary_id": grp["binary_id"],
                "program_name": grp["program_name"],
                "program_index": grp["program_index"],
                "num_devices": grp["num_devices"],
                "min": side(grp["lo"]),
                "max": side(grp["hi"]),
            }
        )

    # worst-first: lowest min pcc at the top (null pcc sinks to the bottom).
    failures.sort(
        key=lambda f: (f["min"] is None, f["min"]["pcc"] if f["min"] else 0.0)
    )

    harness_issues = [
        {
            "op": op,
            "check": check,
            "status": status,
            "count": info["count"],
            "example": info["example"],
        }
        for (op, check, status), info in harness.items()
    ]
    harness_issues.sort(key=lambda h: -h["count"])

    binary_ids = sorted(
        {f["binary_id"] for f in failures if f["binary_id"] is not None}
    )

    summary = {
        "input": args.input,
        "totals": {
            "isolated_numerics_failures": len(failures),
            "accumulated_numerics_failures": accumulated_fail,
            "harness_issues": sum(h["count"] for h in harness_issues),
        },
        "failures": failures,
        "harness_issues": harness_issues,
        "binary_ids_with_failures": binary_ids,
    }
    if skipped_parse:
        summary["skipped_unparseable_lines"] = skipped_parse

    text = json.dumps(summary, indent=2)
    if args.summary_out:
        with open(args.summary_out, "w") as out:
            out.write(text + "\n")
        print(
            f"[done] {len(failures)} isolated failing op(s), "
            f"{len(harness_issues)} harness issue group(s); "
            f"summary -> {args.summary_out}"
            + (f", filtered -> {args.filtered_out}" if args.filtered_out else ""),
            file=sys.stderr,
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

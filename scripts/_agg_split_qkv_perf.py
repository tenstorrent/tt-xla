#!/usr/bin/env python3
"""Aggregate split_qkv ops_perf_results CSV into a per-op summary.

Groups by (OP CODE, output shape) so the input reshape and the query reshape
(both ReshapeViewDeviceOperation) are reported separately.
"""
import csv
import statistics as st
import sys
from collections import OrderedDict


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(csv_path, drop_warmup=10):
    rows = list(csv.DictReader(open(csv_path)))
    groups = OrderedDict()
    for r in rows:
        if num(r.get("DEVICE KERNEL DURATION [ns]")) is None:
            continue  # signpost / non-device row
        osig = "[{},{},{},{}]".format(
            r.get("OUTPUT_0_W_PAD[LOGICAL]", ""),
            r.get("OUTPUT_0_Z_PAD[LOGICAL]", ""),
            r.get("OUTPUT_0_Y_PAD[LOGICAL]", ""),
            r.get("OUTPUT_0_X_PAD[LOGICAL]", ""),
        )
        groups.setdefault((r["OP CODE"], osig), []).append(r)

    lines = ["total op rows: {}".format(len(rows))]
    sum_kernel = 0.0
    sum_fw = 0.0
    for (op, osig), rs in groups.items():
        stable = rs[drop_warmup:] if len(rs) > drop_warmup else rs
        kd = [v for v in (num(r["DEVICE KERNEL DURATION [ns]"]) for r in stable) if v is not None]
        fw = [v for v in (num(r["DEVICE FW DURATION [ns]"]) for r in stable) if v is not None]
        cores = rs[0]["CORE COUNT"]
        lines.append("")
        lines.append("== {}  out_WZYX={}  (count={}, stable={}) ==".format(op, osig, len(rs), len(stable)))
        lines.append(
            "  DEVICE KERNEL DURATION [ns]: mean={:.1f} median={:.1f} min={:.0f} max={:.0f}".format(
                st.mean(kd), st.median(kd), min(kd), max(kd)
            )
        )
        lines.append("  DEVICE FW DURATION [ns]:     mean={:.1f} median={:.1f}".format(st.mean(fw), st.median(fw)))
        lines.append("  CORE COUNT={}".format(cores))
        sum_kernel += st.median(kd)
        sum_fw += st.median(fw)

    lines.append("")
    lines.append("SUM of per-op median DEVICE KERNEL DURATION (one patch iter): {:.1f} ns".format(sum_kernel))
    lines.append("SUM of per-op median DEVICE FW DURATION  (one patch iter):    {:.1f} ns".format(sum_fw))
    out = "\n".join(lines)
    print(out)
    return out


if __name__ == "__main__":
    main(sys.argv[1])

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize TopKDeviceOperation core-count + device time from a tracy ops CSV.

Usage:
    python tests/benchmark/parse_topk_cores.py <ops_perf_results_*.csv>

Used to verify the multi-core distributed-topk fix (#4494): the per-shard local
topk should report CORE COUNT ~17 (~700us); CORE COUNT 1 (~14700us) means the
single-core fallback (fix not built / not active).
"""

import csv
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    rows = [
        r for r in csv.DictReader(open(path)) if r["OP CODE"] == "TopKDeviceOperation"
    ]
    c = Counter(
        (r["CORE COUNT"], round(float(r["DEVICE FW DURATION [ns]"]) / 1000))
        for r in rows
    )
    print(f"{len(rows)} TopKDeviceOperation rows:")
    for (cores, us), n in sorted(c.items()):
        print(f"  cores={cores:>3}  ~{us}us  x{n}")
    verdict = (
        "FIX ACTIVE (multi-core)"
        if any(int(cores) > 1 for cores, _ in c)
        else "single-core fallback (fix NOT active in this build)"
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()

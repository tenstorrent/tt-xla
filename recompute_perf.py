#!/usr/bin/env python3
"""Re-derive perf_report.md using max-per-op across chips (critical-path)
instead of sum-across-chips (chip-seconds).

Uses the prefill window opened after `warmup_complete` (the real measurement
run, not the warmup runs that precede it). If the closing `prefill_end` is
missing from the trace, falls back to the last observed op host timestamp."""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/home/ttuser/agobeljic/tt-xla/results_device")
REPORT = RESULTS / "perf_report.md"

# Parse existing report to get row order
rows = []
for line in REPORT.read_text().splitlines():
    m = re.match(r"\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
    if m:
        model, bs, seq, opt = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        rows.append((model, bs, seq, opt))

def compute(run_dir: Path):
    csvs = list(run_dir.glob("tracy/reports/*/ops_perf_results_*.csv"))
    if not csvs:
        return None
    csv_path = csvs[0]

    signposts = []
    ops = []  # (host_ts, device_id, device_kernel_dur_ns)
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[10]:
                continue
            try:
                ts = int(row[10])
            except ValueError:
                continue
            if row[1] == "signpost":
                signposts.append((row[0], ts))
            else:
                try:
                    did = int(row[3]) if row[3] else -1
                    dev = int(row[18]) if row[18] else 0
                except ValueError:
                    continue
                ops.append((ts, did, dev))

    # measurement window: first prefill_start after warmup_complete,
    # closed by the next prefill_end (or last op host_ts if missing).
    warmup_idx = next(
        (i for i, (n, _) in enumerate(signposts) if "warmup_complete" in n), None
    )
    if warmup_idx is None:
        return None
    s = next(
        (ts for n, ts in signposts[warmup_idx + 1 :] if "prefill_start" in n), None
    )
    if s is None:
        return None
    e = next(
        (ts for n, ts in signposts[warmup_idx + 1 :] if "prefill_end" in n and ts > s),
        None,
    )
    if e is None:
        op_ts_after = [t for t, _, _ in ops if t >= s]
        if not op_ts_after:
            return None
        e = max(op_ts_after)
    ttft_ns = e - s
    in_window = [(did, d) for t, did, d in ops if s <= t <= e]

    # Original (buggy): sum across all (op, chip) rows
    device_sum_ns = sum(d for _, d in in_window)

    # Corrected: sum kernel durations per chip, then take the busiest chip
    per_chip = defaultdict(int)
    for did, d in in_window:
        per_chip[did] += d
    device_max_ns = max(per_chip.values()) if per_chip else 0

    host_max_ns = max(ttft_ns - device_max_ns, 0)

    return ttft_ns, device_sum_ns, device_max_ns, host_max_ns


lines = [
    "| model | bs | seq | opt | ttft_ms | device_ms_sum | device_ms_max | host_ms_max |",
    "|-------|----|-----|-----|---------|---------------|---------------|-------------|",
]
for model, bs, seq, opt in rows:
    run_dir = RESULTS / f"{model}_bs{bs}_seq{seq}_opt{opt}"
    out = compute(run_dir)
    if out is None:
        lines.append(f"| {model} | {bs} | {seq} | {opt} | - | - | - | - |")
        continue
    ttft_ns, dev_sum_ns, dev_max_ns, host_max_ns = out
    lines.append(
        f"| {model} | {bs} | {seq} | {opt} | "
        f"{ttft_ns/1e6:.1f} | {dev_sum_ns/1e6:.1f} | {dev_max_ns/1e6:.1f} | {host_max_ns/1e6:.1f} |"
    )

REPORT.write_text("\n".join(lines) + "\n")
print(f"Wrote {REPORT}")

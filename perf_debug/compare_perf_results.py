#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare sampling perf results from two run_sampling_perf_suite.sh output dirs.

Usage:
    python perf_debug/compare_perf_results.py <without_fix_dir> <with_fix_dir>

Example:
    python perf_debug/compare_perf_results.py perf_mar31_no_fix perf_mar31_with_fix
"""

import os
import re
import sys


def tail_lines(path, n=500):
    """Read last n lines of a file efficiently."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        buf = bytearray()
        pos = size
        lines_found = 0
        chunk = 8192
        while pos > 0 and lines_found < n:
            read = min(chunk, pos)
            pos -= read
            f.seek(pos)
            buf[:0] = f.read(read)
            lines_found = buf.count(b"\n")
        text = buf.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n:])


def read_head(path, n=30):
    """Read first n lines of a file."""
    with open(path, errors="replace") as f:
        return "".join(f.readline() for _ in range(n))


def parse_wall_time(log_path):
    """Extract total wall-clock time (seconds) for the whole test run.

    - chat.py logs: diff between first and last timestamp in the file.
    - pytest logs: parse "X passed in Y.Ys" from the summary line.
    """
    if not os.path.exists(log_path):
        return None

    # pytest format: "1 passed in 123.12s (0:02:03)"
    tail = tail_lines(log_path, 20)
    m = re.search(r"passed.*? in ([\d.]+)s", tail)
    if m:
        return float(m.group(1))

    # chat.py / vLLM log format: "INFO MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    # Scan first 30 lines for start, last 250 lines for end timestamp.
    head = read_head(log_path, 30)
    shallow_tail = tail_lines(log_path, 250)
    patterns = [
        (r"\d{2}-\d{2} (\d{2}:\d{2}:\d{2})", "%H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})", "%H:%M:%S"),
    ]
    for ts_pat, ts_fmt in patterns:
        m_start = re.search(ts_pat, head)
        m_end = None
        for line in shallow_tail.splitlines():
            m = re.search(ts_pat, line)
            if m:
                m_end = m
        if m_start and m_end:
            from datetime import datetime
            try:
                t0 = datetime.strptime(m_start.group(1), ts_fmt)
                t1 = datetime.strptime(m_end.group(1), ts_fmt)
                diff = (t1 - t0).total_seconds()
                if diff < 0:
                    diff += 86400  # midnight wrap
                if diff > 0:
                    return diff
            except ValueError:
                continue

    return None


def parse_metrics(log_path):
    """Extract (tok_s, wall_time_s) from a log file."""
    if not os.path.exists(log_path):
        return None, None
    text = tail_lines(log_path, 500)

    tok_s = None
    wall_time = parse_wall_time(log_path)

    # chat.py benchmark format
    m = re.search(r"Overall tok/s:\s+([\d.]+)", text)
    if m:
        return float(m.group(1)), wall_time

    # pytest vllm_benchmark format: "decode_tps=21.5"
    tps_vals = re.findall(r"decode_tps=([\d.]+)", text)
    if tps_vals:
        tok_s = sum(float(v) for v in tps_vals) / len(tps_vals)

    return tok_s, wall_time


def detect_sampling_from_log(log_path):
    """Detect (sampling, on_device) from log content."""
    if not os.path.exists(log_path):
        return "non-greedy", True
    head = read_head(log_path, 60)
    cpu = "cpu_sampling': True" in head or "cpu_sampling=True" in head
    # vllm-bench uses ignore_eos=True with temperature=0 (greedy); chat.py logs temperature explicitly
    greedy = ("ignore_eos=True" in head or "temperature=0.0" in head
              or "temperature': 0.0" in head or "temperature': 0}" in head)
    sampling = "greedy" if greedy else "non-greedy"
    on_device = not cpu
    return sampling, on_device


def parse_log_name(name, log_path=None):
    """Parse log filename into (model, sampling, seq_len, batch_size, on_device).

    Returns dict or None if unrecognized.
    """
    # vllm_bench pytest format: vllm_bench_llama3p2_1b_batch1
    m = re.match(r"vllm_bench_(llama3p[12]_\w+)_(batch\d+)", name)
    if m:
        model_raw, batch_raw = m.group(1), m.group(2)
        batch = int(batch_raw.replace("batch", ""))
        model = _fmt_model(model_raw)
        sampling, on_device = detect_sampling_from_log(log_path)
        return dict(
            model=model,
            sampling=sampling,
            seq_len=128,
            batch=batch,
            on_device=on_device,
            harness="vllm-bench",
            label=name,
        )

    # chat.py format: {model}_{seq}_{batch?}_{sampling}.log
    # e.g. llama3p1_8b_seq128_non_greedy_device
    #      opt125m_seq2048_batch32_non_greedy_device
    m = re.match(
        r"(opt125m|llama3p[12]_\w+b)_seq(\d+)(?:_batch(\d+))?_(greedy_device|non_greedy_cpu|non_greedy_device)",
        name,
    )
    if not m:
        return None

    model_raw = m.group(1)
    seq_len = int(m.group(2))
    batch = int(m.group(3)) if m.group(3) else 1
    sampling_raw = m.group(4)

    sampling_map = {
        "greedy_device": "greedy",
        "non_greedy_cpu": "non-greedy",
        "non_greedy_device": "non-greedy",
    }
    on_device = sampling_raw != "non_greedy_cpu"

    return dict(
        model=_fmt_model(model_raw),
        sampling=sampling_map[sampling_raw],
        seq_len=seq_len,
        batch=batch,
        on_device=on_device,
        harness="chat.py",
        label=name,
    )


def _fmt_model(raw):
    mapping = {
        "opt125m": "OPT-125M",
        "llama3p2_1b": "Llama-3.2-1B",
        "llama3p2_3b": "Llama-3.2-3B",
        "llama3p1_8b": "Llama-3.1-8B",
    }
    return mapping.get(raw, raw)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <without_fix_dir> <with_fix_dir>")
        sys.exit(1)

    no_fix_dir = sys.argv[1]
    fix_dir = sys.argv[2]

    # Collect all log names from without-fix dir
    if not os.path.isdir(no_fix_dir):
        print(f"Directory not found: {no_fix_dir}")
        sys.exit(1)

    log_names = sorted(
        f[:-4] for f in os.listdir(no_fix_dir) if f.endswith(".log")
    )

    rows = []
    seen = set()  # deduplicate by (model, sampling, seq_len, batch, on_device)
    for name in log_names:
        log_path = os.path.join(no_fix_dir, name + ".log")
        meta = parse_log_name(name, log_path)
        if meta is None:
            continue

        key = (meta["model"], meta["sampling"], meta["seq_len"], meta["batch"], meta["on_device"], meta["harness"])
        if key in seen:
            continue
        seen.add(key)

        no_fix_val, no_fix_time = parse_metrics(os.path.join(no_fix_dir, name + ".log"))
        fix_val, _ = parse_metrics(os.path.join(fix_dir, name + ".log"))

        if no_fix_val is not None and fix_val is not None and no_fix_val > 0:
            pct = (fix_val - no_fix_val) / no_fix_val * 100
        else:
            pct = None

        rows.append({**meta, "no_fix": no_fix_val, "fix": fix_val, "pct": pct, "time": no_fix_time})

    # Sort: model → sampling → seq_len → batch
    model_order = {"OPT-125M": 0, "Llama-3.2-1B": 1, "Llama-3.2-3B": 2, "Llama-3.1-8B": 3}
    # Sort: harness first, then model → sampling → seq_len → batch
    rows.sort(
        key=lambda r: (
            r["harness"],
            model_order.get(r["model"], 99),
            r["sampling"],
            r["on_device"],
            r["seq_len"],
            r["batch"],
        )
    )

    # Print table
    col_widths = [14, 12, 8, 6, 9, 11, 10, 10, 9, 12]
    headers = ["Model", "Sampling", "SeqLen", "Batch", "OnDevice", "WithoutFix", "WithFix", "Change", "Time(s)", "Harness"]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(
        f" {h:<{w}} " for h, w in zip(headers, col_widths)
    ) + "|"

    print()
    print(f"  {no_fix_dir}  vs  {fix_dir}")
    print()
    print(sep)
    print(header_row)
    print(sep)

    prev_model = None
    prev_harness = None
    for r in rows:
        if r["harness"] != prev_harness and prev_harness is not None:
            print(sep)
            print(sep)
        elif r["model"] != prev_model and prev_model is not None:
            print(sep)
        prev_model = r["model"]
        prev_harness = r["harness"]

        no_fix_s = f"{r['no_fix']:.2f}" if r["no_fix"] is not None else "N/A"
        fix_s = f"{r['fix']:.2f}" if r["fix"] is not None else "N/A"

        if r["pct"] is not None:
            sign = "+" if r["pct"] >= 0 else ""
            pct_s = f"{sign}{r['pct']:.1f}%"
        else:
            pct_s = "N/A"

        on_dev_s = "Yes" if r["on_device"] else "CPU"
        time_s = f"{r['time']:.0f}s" if r["time"] is not None else "N/A"
        vals = [
            r["model"], r["sampling"], str(r["seq_len"]), str(r["batch"]),
            on_dev_s, no_fix_s, fix_s, pct_s, time_s, r["harness"],
        ]
        print("|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, col_widths)) + "|")

    print(sep)
    print()

    # Summary: only rows where fix matters (non-greedy device)
    device_rows = [r for r in rows if r["on_device"] and r["sampling"] == "non-greedy" and r["pct"] is not None]
    if device_rows:
        avg_pct = sum(r["pct"] for r in device_rows) / len(device_rows)
        print(f"  Average improvement (non-greedy device): {avg_pct:+.1f}%")
        print()


if __name__ == "__main__":
    main()

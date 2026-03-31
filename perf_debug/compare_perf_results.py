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


def parse_tok_s(log_path):
    """Extract tok/s from a log file. Returns None if not found."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        text = f.read()

    # chat.py benchmark format
    m = re.search(r"Overall tok/s:\s+([\d.]+)", text)
    if m:
        return float(m.group(1))

    # pytest vllm_benchmark format: "decode_tps=21.5"
    # Use average of all requests' decode_tps
    tps_vals = re.findall(r"decode_tps=([\d.]+)", text)
    if tps_vals:
        return sum(float(v) for v in tps_vals) / len(tps_vals)

    return None


def parse_log_name(name):
    """Parse log filename into (model, sampling, seq_len, batch_size, on_device).

    Returns dict or None if unrecognized.
    """
    # vllm_bench pytest format: vllm_bench_llama3p2_1b_batch1
    m = re.match(r"vllm_bench_(llama3p[12]_\w+)_(batch\d+)", name)
    if m:
        model_raw, batch_raw = m.group(1), m.group(2)
        batch = int(batch_raw.replace("batch", ""))
        model = _fmt_model(model_raw)
        return dict(
            model=model,
            sampling="non-greedy",
            seq_len=128,
            batch=batch,
            on_device=True,
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
        meta = parse_log_name(name)
        if meta is None:
            continue

        key = (meta["model"], meta["sampling"], meta["seq_len"], meta["batch"], meta["on_device"])
        if key in seen:
            continue
        seen.add(key)

        no_fix_val = parse_tok_s(os.path.join(no_fix_dir, name + ".log"))
        fix_val = parse_tok_s(os.path.join(fix_dir, name + ".log"))

        if no_fix_val is not None and fix_val is not None and no_fix_val > 0:
            pct = (fix_val - no_fix_val) / no_fix_val * 100
        else:
            pct = None

        rows.append({**meta, "no_fix": no_fix_val, "fix": fix_val, "pct": pct})

    # Sort: model → sampling → seq_len → batch
    model_order = {"OPT-125M": 0, "Llama-3.2-1B": 1, "Llama-3.2-3B": 2, "Llama-3.1-8B": 3}
    rows.sort(
        key=lambda r: (
            model_order.get(r["model"], 99),
            r["sampling"],
            r["on_device"],
            r["seq_len"],
            r["batch"],
        )
    )

    # Print table
    col_widths = [14, 12, 8, 6, 9, 11, 10, 10]
    headers = ["Model", "Sampling", "SeqLen", "Batch", "OnDevice", "WithoutFix", "WithFix", "Change"]

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
    for r in rows:
        if r["model"] != prev_model and prev_model is not None:
            print(sep)
        prev_model = r["model"]

        no_fix_s = f"{r['no_fix']:.2f}" if r["no_fix"] is not None else "N/A"
        fix_s = f"{r['fix']:.2f}" if r["fix"] is not None else "N/A"

        if r["pct"] is not None:
            sign = "+" if r["pct"] >= 0 else ""
            pct_s = f"{sign}{r['pct']:.1f}%"
        else:
            pct_s = "N/A"

        on_dev_s = "Yes" if r["on_device"] else "CPU"
        vals = [
            r["model"], r["sampling"], str(r["seq_len"]), str(r["batch"]),
            on_dev_s, no_fix_s, fix_s, pct_s,
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

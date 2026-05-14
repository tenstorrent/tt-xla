#!/usr/bin/env python3
"""Compute TOP1/TOP5 accuracy from a run_codegen_decode.py log + .refpt reference.

Reads AUTORESEARCH_GRAPH_*_PERPOS_ARGMAX_BATCH0 + _PERPOS_TOP5_BATCH0 lines from
a harness stdout log, compares against the matched reference saved in the
.refpt file, prints accuracy in the same format as the pytest path:

    Token accuracy: TOP1=XX.XX%, TOP5=YY.YY%

Usage:
    python codegen_accuracy_from_log.py \\
        --log autoresearch_logs/codegen_1lyr_fresh.log \\
        --refpt tests/benchmark/reference_outputs/gpt-oss-120b-1layer.refpt \\
        --graph graph_0
"""
import argparse
import re
import sys

import torch


def parse_argmax_line(text, key):
    """Return list[int] parsed from 'KEY=v1,v2,v3,...' line, or None."""
    pat = re.compile(rf"^{re.escape(key)}=([0-9,\-]+)$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    return [int(x) for x in m.group(1).split(",")]


def parse_top5_line(text, key):
    """Return list[list[int]] parsed from 'KEY=v;v;v...' line, or None."""
    pat = re.compile(rf"^{re.escape(key)}=([0-9,;\-]+)$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    rows = []
    for row in m.group(1).split(";"):
        rows.append([int(x) for x in row.split(",")])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Path to harness stdout log file")
    p.add_argument("--refpt", required=True, help="Path to .refpt reference file")
    p.add_argument(
        "--graph",
        default="graph_0",
        help="Graph name to extract predictions from (default: graph_0 = logits prefill)",
    )
    args = p.parse_args()

    log = open(args.log).read()
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    refdata = torch.load(args.refpt)
    reference_tokens = refdata["reference_tokens"]  # [1, total_length]
    full_top1_tokens = refdata["top1_tokens"]  # [total_length-1]
    full_top5_tokens = refdata["top5_tokens"]  # [total_length-1, 5]
    total_length = reference_tokens.shape[-1]
    split_point = total_length // 2

    graph_upper = args.graph.upper()
    pred_argmax = parse_argmax_line(log, f"AUTORESEARCH_{graph_upper}_PERPOS_ARGMAX_BATCH0")
    pred_top5 = parse_top5_line(log, f"AUTORESEARCH_{graph_upper}_PERPOS_TOP5_BATCH0")
    if pred_argmax is None:
        sys.exit(
            f"No AUTORESEARCH_{graph_upper}_PERPOS_ARGMAX_BATCH0 line found in {args.log}"
        )

    print(f"[codegen-accuracy] reference total_length={total_length}, split_point={split_point}")
    print(f"[codegen-accuracy] TT predicted (per-position argmax for batch 0): n={len(pred_argmax)}")

    # For prefill graph: pred_argmax[i] = TT's prediction for token at position i+1.
    # Compare against full_top1_tokens[0..len(pred_argmax)-1].
    n = min(len(pred_argmax), full_top1_tokens.shape[0])
    cpu_top1 = [int(full_top1_tokens[i].item()) for i in range(n)]
    cpu_top5 = [full_top5_tokens[i].tolist() for i in range(n)]
    tt_argmax = pred_argmax[:n]

    top1_hits = sum(1 for i in range(n) if tt_argmax[i] == cpu_top1[i])
    top5_hits = sum(1 for i in range(n) if tt_argmax[i] in cpu_top5[i])
    top1 = 100.0 * top1_hits / n
    top5 = 100.0 * top5_hits / n
    print(f"Token accuracy: TOP1={top1:.2f}%, TOP5={top5:.2f}%  (over {n} positions)")

    mismatches = [
        (i, tt_argmax[i], cpu_top1[i]) for i in range(n) if tt_argmax[i] != cpu_top1[i]
    ]
    print(f"[codegen-accuracy] {len(mismatches)}/{n} positions differ from CPU top1")
    if mismatches:
        print(f"[codegen-accuracy] first 8 mismatches (pos, tt, cpu_top1): {mismatches[:8]}")


if __name__ == "__main__":
    main()

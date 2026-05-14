#!/usr/bin/env python3
"""Inject the missing attention_sink kwarg into prefill SDPA calls.

The codegen-emitted prefill main.py is missing the `attention_sink` kwarg
on every `ttnn.transformer.scaled_dot_product_attention` call. The decode
main.py (same artifact, neighboring graph dir) already passes it via
`ttnn.transformer.scaled_dot_product_attention_decode`. Per-layer the
weight is the same — same const_eval expression on both sides.

This script extracts the attention_sink expression from each decode SDPA
call in order, then injects it into the corresponding prefill SDPA call
in order. The Nth prefill call gets the Nth decode call's expression.

Usage:
    python patch_attention_sink.py \\
        --prefill <artifact>/graph_0/main.py \\
        --decode  <artifact>/graph_1/main.py
"""
import argparse
import re
import shutil
import sys


SDPA_PREFILL_PAT = re.compile(r"ttnn\.transformer\.scaled_dot_product_attention\($")
SDPA_DECODE_PAT = re.compile(r"ttnn\.transformer\.scaled_dot_product_attention_decode\($")
ATTENTION_SINK_PAT = re.compile(r"^\s*attention_sink=(.+),$")
IS_CAUSAL_LINE_PAT = re.compile(r"^(\s*)is_causal=")


def find_calls_with_close(lines, header_pat):
    """Find call sites: list of (header_line_idx, close_line_idx).
    Header line is the one matching header_pat; close line is the matching `)`."""
    out = []
    for i, line in enumerate(lines):
        if header_pat.search(line):
            # walk forward to find matching close
            depth = line.count("(") - line.count(")")
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += lines[j].count("(") - lines[j].count(")")
            if depth == 0:
                out.append((i, j))
    return out


def extract_attention_sink_expressions(decode_lines):
    """For each SDPA-decode call in decode_lines, return its attention_sink= value."""
    calls = find_calls_with_close(decode_lines, SDPA_DECODE_PAT)
    exprs = []
    for start, end in calls:
        body = decode_lines[start:end + 1]
        sink_expr = None
        for ln in body:
            m = ATTENTION_SINK_PAT.match(ln)
            if m:
                sink_expr = m.group(1)
                break
        if sink_expr is None:
            raise RuntimeError(
                f"No attention_sink= kwarg found in decode SDPA call at line "
                f"{start + 1}..{end + 1}"
            )
        exprs.append(sink_expr)
    return exprs


def inject_attention_sink(prefill_lines, sink_exprs):
    """Insert attention_sink=<expr>, into each prefill SDPA call body, after
    the is_causal= line so the formatting matches the colleague's diff style."""
    calls = find_calls_with_close(prefill_lines, SDPA_PREFILL_PAT)
    if len(calls) != len(sink_exprs):
        raise RuntimeError(
            f"Call count mismatch: {len(calls)} prefill SDPA calls but "
            f"{len(sink_exprs)} decode attention_sink expressions"
        )

    # Patch in reverse so line indices stay stable.
    out = list(prefill_lines)
    patches_applied = 0
    for (start, end), expr in reversed(list(zip(calls, sink_exprs))):
        # Sanity: skip if this SDPA call already has attention_sink (idempotent).
        already_has = any(
            ATTENTION_SINK_PAT.match(ln) for ln in out[start:end + 1]
        )
        if already_has:
            continue
        # Find the is_causal= line in this call's body and inject right after.
        injected = False
        for j in range(start, end + 1):
            m = IS_CAUSAL_LINE_PAT.match(out[j])
            if m:
                indent = m.group(1)
                new_line = f"{indent}attention_sink={expr},\n"
                out.insert(j + 1, new_line)
                patches_applied += 1
                injected = True
                break
        if not injected:
            raise RuntimeError(
                f"Could not find is_causal= line in prefill SDPA call at "
                f"line {start + 1}..{end + 1}"
            )
    return out, patches_applied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefill", required=True, help="Path to prefill main.py")
    p.add_argument("--decode", required=True, help="Path to decode main.py")
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .orig backup of the prefill file",
    )
    args = p.parse_args()

    with open(args.decode) as f:
        decode_lines = f.readlines()
    with open(args.prefill) as f:
        prefill_lines = f.readlines()

    sink_exprs = extract_attention_sink_expressions(decode_lines)
    print(f"[patch] extracted {len(sink_exprs)} attention_sink expressions from decode")
    for i, e in enumerate(sink_exprs[:3]):
        print(f"[patch]   layer {i}: {e}")
    if len(sink_exprs) > 3:
        print(f"[patch]   ... and {len(sink_exprs) - 3} more")

    if not args.no_backup:
        shutil.copy(args.prefill, args.prefill + ".orig")
        print(f"[patch] backup: {args.prefill}.orig")

    patched_lines, n = inject_attention_sink(prefill_lines, sink_exprs)
    with open(args.prefill, "w") as f:
        f.writelines(patched_lines)
    print(f"[patch] injected {n} attention_sink kwargs into {args.prefill}")

    # Verify
    with open(args.prefill) as f:
        text = f.read()
    new_sink_count = text.count("attention_sink=")
    print(f"[patch] post-patch attention_sink count: {new_sink_count}")


if __name__ == "__main__":
    main()

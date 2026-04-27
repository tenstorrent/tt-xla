#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analyze a tracy ops_perf_results.csv from sampler/vLLM perf runs.

Common usage:
    # Hotspot table (top ops by total device fw duration, warmups skipped)
    python perf_debug/analyze_tracy.py path/to/ops_perf_results.csv

    # Compare two runs (baseline vs new)
    python perf_debug/analyze_tracy.py baseline.csv --vs new.csv

    # Filter to a single signpost block (e.g. "iter 3")
    python perf_debug/analyze_tracy.py path/to/csv --signpost "iter 3"

    # Per-iter timing breakdown (one row per signpost)
    python perf_debug/analyze_tracy.py path/to/csv --per-iter

    # Forward/sampling/other category split
    python perf_debug/analyze_tracy.py path/to/csv --categorize

    # Drill into a single op (per-call distribution)
    python perf_debug/analyze_tracy.py path/to/csv --detail "Matmul"

Notes:
    - Warmups (signposts with "warmup" in the label) are skipped by default.
      Pass --include-warmup to include them.
    - The duration column defaults to "DEVICE FW DURATION [ns]" (firmware-side
      total, includes both kernel and dispatch). Pass --duration "DEVICE KERNEL
      DURATION [ns]" for kernel-only.
    - Categorization uses substring matching on OP CODE. "softmax" is grouped
      under sampling; if you're analyzing a vLLM forward+sampling trace, expect
      attention-softmax to be miscategorized — use --detail to drill in.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Substring matching, case-insensitive. Order matters within a category but
# not between categories — first matching list wins (sampling checked first
# so that sampling-side softmax wins over the forward "softmax" entry).
SAMPLING_OPS = (
    "sort",
    "topk",
    "argmax",
    "scatter",
    "gather",
    "tt::sampling",
    "sampling",
    "log_softmax",
    "softmax",
)
FORWARD_OPS = (
    "matmul",
    "linear",
    "embedding",
    "rotary",
    "paged_cache",
    "scaled_dot_product_attention",
    "attention",
    "layernorm",
    "rms_norm",
    "rmsnorm",
    "silu",
    "swiglu",
    "gelu",
)
LAYOUT_OPS = (
    "tilize",
    "untilize",
    "typecast",
    "to_layout",
    "to_memory_config",
    "permute",
    "reshape",
    "transpose",
    "pad",
)


def categorize(op_code: str) -> str:
    name = op_code.lower()
    for s in SAMPLING_OPS:
        if s in name:
            return "sampling"
    for f in FORWARD_OPS:
        if f in name:
            return "forward"
    for l in LAYOUT_OPS:
        if l in name:
            return "layout"
    return "other"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "HOST DURATION [ns]",
        "HOST START TS",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def signpost_indices(df: pd.DataFrame) -> list[int]:
    return df.index[df["OP TYPE"] == "signpost"].tolist()


def trim_warmup(df: pd.DataFrame) -> pd.DataFrame:
    sp = signpost_indices(df)
    if not sp:
        return df
    last_warmup = -1
    for i, idx in enumerate(sp):
        if "warmup" in str(df.at[idx, "OP CODE"]).lower():
            last_warmup = i
    if last_warmup == -1:
        return df
    start = sp[last_warmup] + 1
    # reset_index so downstream signpost_indices + iloc slicing align
    return df.iloc[start:].reset_index(drop=True)


def filter_by_signpost(df: pd.DataFrame, substr: str) -> pd.DataFrame:
    sp = signpost_indices(df)
    matches = [
        i for i in sp if substr.lower() in str(df.at[i, "OP CODE"]).lower()
    ]
    if not matches:
        print(f"No signposts matching {substr!r}", file=sys.stderr)
        return df.iloc[0:0]
    start = matches[0]
    after = [s for s in sp if s > start]
    end = after[0] if after else len(df)
    return df.iloc[start + 1 : end].reset_index(drop=True)


def ops_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["OP TYPE"] != "signpost"]


def timed_signposts(df: pd.DataFrame) -> list[int]:
    """Signpost indices excluding warmup-labeled ones."""
    return [
        i
        for i in signpost_indices(df)
        if "warmup" not in str(df.at[i, "OP CODE"]).lower()
    ]


def slice_last_n_iters(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
    """Slice df to ops within the last n timed iter signpost ranges.
    Returns (sliced_df, actual_n_taken). Empty df + 0 if no timed signposts."""
    sp = timed_signposts(df)
    if not sp:
        return df.iloc[0:0].reset_index(drop=True), 0
    take = min(n, len(sp))
    start = sp[-take] + 1
    return df.iloc[start:].reset_index(drop=True), take


def hotspot_table(df: pd.DataFrame, top: int, dur_col: str) -> tuple:
    """Return (table, total_ms, total_count, n_distinct) for printing."""
    ops = ops_only(df)
    if ops.empty or dur_col not in ops.columns:
        return pd.DataFrame(), 0.0, 0, 0
    total_ms = ops[dur_col].sum() / 1e6
    total_count = len(ops)
    g = ops.groupby("OP CODE")[dur_col].agg(["sum", "count", "mean"])
    g["sum_ms"] = g["sum"] / 1e6
    g["mean_us"] = g["mean"] / 1e3
    g["pct"] = 100 * g["sum"] / g["sum"].sum()
    n_distinct = len(g)
    return (
        g.sort_values("sum", ascending=False).head(top)[
            ["sum_ms", "count", "mean_us", "pct"]
        ],
        total_ms,
        total_count,
        n_distinct,
    )


def category_table(df: pd.DataFrame, dur_col: str) -> pd.DataFrame:
    ops = ops_only(df).copy()
    if ops.empty or dur_col not in ops.columns:
        return pd.DataFrame()
    ops["category"] = ops["OP CODE"].astype(str).map(categorize)
    g = ops.groupby("category")[dur_col].agg(["sum", "count"])
    g["sum_ms"] = g["sum"] / 1e6
    g["pct"] = 100 * g["sum"] / g["sum"].sum()
    return g[["count", "sum_ms", "pct"]].sort_values("sum_ms", ascending=False)


def per_iter_table(df: pd.DataFrame, dur_col: str) -> pd.DataFrame:
    sp = signpost_indices(df)
    rows = []
    for i, idx in enumerate(sp):
        label = str(df.at[idx, "OP CODE"])
        start = idx + 1
        end = sp[i + 1] if i + 1 < len(sp) else len(df)
        ops = ops_only(df.iloc[start:end])
        total_ns = ops[dur_col].sum() if dur_col in ops.columns else 0
        rows.append(
            {"signpost": label, "n_ops": len(ops), "total_ms": total_ns / 1e6}
        )
    return pd.DataFrame(rows)


def detail_op(df: pd.DataFrame, op_substr: str, dur_col: str) -> pd.DataFrame:
    ops = ops_only(df)
    matches = ops[ops["OP CODE"].astype(str).str.contains(op_substr, case=False)]
    if matches.empty or dur_col not in matches.columns:
        return pd.DataFrame()
    g = matches.groupby("OP CODE")[dur_col].describe(
        percentiles=[0.1, 0.5, 0.9]
    )
    # convert ns → us for readability
    for col in ("mean", "std", "min", "10%", "50%", "90%", "max"):
        if col in g.columns:
            g[col] = g[col] / 1e3
    g = g.rename(columns={c: f"{c}_us" for c in g.columns if c != "count"})
    return g.sort_values("count", ascending=False)


def diff_table(
    base: pd.DataFrame, new: pd.DataFrame, top: int, dur_col: str
) -> tuple:
    """Return (table, base_total_ms, new_total_ms) for printing."""
    base_ops = ops_only(base)
    new_ops = ops_only(new)
    base_total = base_ops[dur_col].sum() / 1e6
    new_total = new_ops[dur_col].sum() / 1e6
    b = base_ops.groupby("OP CODE")[dur_col].sum() / 1e6
    n = new_ops.groupby("OP CODE")[dur_col].sum() / 1e6
    cmp = pd.DataFrame({"base_ms": b, "new_ms": n}).fillna(0)
    cmp["delta_ms"] = cmp["new_ms"] - cmp["base_ms"]
    cmp["delta_pct"] = 100 * cmp["delta_ms"] / cmp["base_ms"].replace(0, pd.NA)
    cmp = cmp.reindex(
        cmp["delta_ms"].abs().sort_values(ascending=False).index
    ).head(top)
    return cmp, base_total, new_total


def main():
    ap = argparse.ArgumentParser(
        description="Analyze tracy ops_perf_results CSV from sampler/vLLM runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("csv", type=Path, help="Path to ops_perf_results.csv")
    ap.add_argument(
        "--vs",
        type=Path,
        default=None,
        help="Compare against this CSV (first arg is baseline)",
    )
    ap.add_argument(
        "--signpost",
        type=str,
        default=None,
        help="Filter to ops between signpost matching SUBSTR and the next signpost",
    )
    ap.add_argument(
        "--per-iter",
        action="store_true",
        help="Show per-signpost iter breakdown (one row per iter)",
    )
    ap.add_argument(
        "--categorize",
        action="store_true",
        help="Show forward/sampling/layout/other category split",
    )
    ap.add_argument(
        "--detail",
        type=str,
        default=None,
        help="Drill into a single op (substring match), show per-call distribution",
    )
    ap.add_argument(
        "--top", type=int, default=20, help="Top N rows in hotspot/diff tables"
    )
    ap.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Default mode: aggregate the last N timed iters (default: 5). "
        "Also prints last-iter-only table when N>1.",
    )
    ap.add_argument(
        "--include-warmup",
        action="store_true",
        help='Include warmup signposts (default: skip rows before last "warmup" signpost)',
    )
    ap.add_argument(
        "--duration",
        type=str,
        default="DEVICE FW DURATION [ns]",
        help='Duration column (default: "DEVICE FW DURATION [ns]")',
    )
    args = ap.parse_args()

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")

    df = load_csv(args.csv)
    n_sp = len(signpost_indices(df))
    print(f"Loaded {args.csv} ({len(df)} rows, {n_sp} signposts)")

    if args.signpost:
        df = filter_by_signpost(df, args.signpost)
        print(f"After --signpost {args.signpost!r}: {len(df)} rows")
    elif not args.include_warmup:
        before = len(df)
        df = trim_warmup(df)
        if len(df) != before:
            print(f"Skipped warmup signposts: {before - len(df)} rows trimmed")

    if args.vs:
        new = load_csv(args.vs)
        if not args.include_warmup:
            new = trim_warmup(new)
        print(f"\n=== Diff: {args.csv.name} → {args.vs.name} (top {args.top} by |Δms|) ===")
        diff, base_tot, new_tot = diff_table(df, new, args.top, args.duration)
        print(diff.to_string())
        delta = new_tot - base_tot
        delta_pct = 100 * delta / base_tot if base_tot else 0
        print(
            f"\nTotal: base={base_tot:.3f} ms, new={new_tot:.3f} ms, "
            f"Δ={delta:+.3f} ms ({delta_pct:+.2f}%)"
        )
        return

    if args.per_iter:
        print(f"\n=== Per-iter breakdown ({args.duration}) ===")
        print(per_iter_table(df, args.duration).to_string(index=False))
        return

    if args.detail:
        print(f"\n=== Detail for ops matching {args.detail!r} ({args.duration}) ===")
        d = detail_op(df, args.detail, args.duration)
        if d.empty:
            print("(no matching ops)")
        else:
            print(d.to_string())
        return

    def _print_hotspot(scope_df: pd.DataFrame, header: str) -> None:
        print(f"\n=== {header} ===")
        table, total_ms, total_count, n_distinct = hotspot_table(
            scope_df, args.top, args.duration
        )
        if table.empty:
            print("(no ops)")
            return
        print(table.to_string())
        shown_pct = table["pct"].sum()
        print(
            f"\nTotal: {total_ms:.3f} ms across {total_count} ops "
            f"({n_distinct} distinct op types; top {len(table)} shown = {shown_pct:.1f}%)"
        )

    if args.categorize:
        print(f"\n=== Category split ({args.duration}) ===")
        print(category_table(df, args.duration).to_string())

    # Default mode: per-iter timings for the last N iters (sanity-check
    # iter-to-iter consistency, spot outliers) + op-level hotspot for the
    # last iter only (representative steady-state breakdown).
    # If --signpost was used, df is already a single iter slice — fall back
    # to one hotspot table.
    timed_sp = timed_signposts(df)
    if args.signpost or not timed_sp:
        _print_hotspot(df, f"Hotspots (top {args.top} by {args.duration})")
        return

    per_iter = per_iter_table(df, args.duration).tail(args.last_n)
    take_n = len(per_iter)
    print(
        f"\n=== Per-iter timing: last {take_n} iter{'s' if take_n != 1 else ''} "
        f"({args.duration}) ==="
    )
    print(per_iter.to_string(index=False))

    last_1_df, _ = slice_last_n_iters(df, 1)
    _print_hotspot(
        last_1_df,
        f"Hotspots: last iter only (top {args.top} by {args.duration})",
    )


if __name__ == "__main__":
    main()

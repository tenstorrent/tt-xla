#!/usr/bin/env python3
"""Plot per-op device kernel time for each results subfolder that has a tracy report."""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("./results1")
DUR_COL = "DEVICE KERNEL DURATION [ns]"
OP_COL = "OP CODE"


def load_prefill_pass(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    starts = df.index[df[OP_COL] == "prefill_start"].tolist()
    ends = df.index[df[OP_COL] == "prefill_end"].tolist()
    if not starts or not ends:
        return pd.DataFrame()
    # Use the last pass to skip warmup
    start, end = starts[-1] + 1, ends[-1]
    return df.iloc[start:end]


def plot_folder(run_dir: Path, csv_path: Path):
    df = load_prefill_pass(csv_path)
    if df.empty or DUR_COL not in df.columns:
        print(f"  skipping {run_dir.name}: no prefill data")
        return

    df[DUR_COL] = pd.to_numeric(df[DUR_COL], errors="coerce").fillna(0)

    per_op = (
        df.groupby(OP_COL)[DUR_COL]
        .sum()
        .sort_values(ascending=False)
    )
    total_ms = per_op.sum() / 1e6

    fig, ax = plt.subplots(figsize=(max(10, len(per_op) * 0.4), 5))
    bars = ax.bar(range(len(per_op)), per_op.values / 1e6)
    ax.set_xticks(range(len(per_op)))
    ax.set_xticklabels(per_op.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Device kernel time (ms)")
    ax.set_title(f"{run_dir.name}\nTotal: {total_ms:.2f} ms")
    ax.bar_label(bars, labels=[f"{v/1e6:.2f}" for v in per_op.values], fontsize=7, padding=2)
    fig.tight_layout()

    out = run_dir / "device_perf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.relative_to(RESULTS_DIR)}  (total {total_ms:.2f} ms)")


def main():
    csvs = sorted(glob.glob(str(RESULTS_DIR / "*/tracy/reports/*/ops_perf_results_*.csv")))
    if not csvs:
        print("No ops_perf_results CSVs found. Run the matrix with tracy first.")
        return

    for csv_path in csvs:
        csv_path = Path(csv_path)
        run_dir = csv_path.parents[3]  # results/<name>/tracy/reports/<ts>/file.csv
        print(run_dir.name)
        plot_folder(run_dir, csv_path)


if __name__ == "__main__":
    main()

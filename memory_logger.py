#!/usr/bin/env python3
# pip install psutil matplotlib
# python memory_logger.py --pid 12345 --interval 0.1 --csv rss.csv --png rss.png

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import psutil


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Log RSS for a specific PID and plot it.")
    parser.add_argument("--pid", type=int, required=True, help="Target process ID")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds")
    parser.add_argument("--csv", type=Path, default=Path("pid_rss.csv"), help="CSV output path")
    parser.add_argument("--png", type=Path, default=Path("pid_rss.png"), help="PNG output path")
    return parser.parse_args()


def generate_plot(csv_path: Path, png_path: Path):
    timestamps = []
    rss_mib = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row["timestamp_utc"]))
            rss_mib.append(float(row["rss_mib"]))

    if not timestamps:
        print("No samples collected, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timestamps, rss_mib, linewidth=2, label="RSS (MiB)")
    ax.set_title("Process RSS Over Time")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("RSS (MiB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {png_path}")


def main():
    args = parse_args()

    try:
        proc = psutil.Process(args.pid)
        proc_name = proc.name()
    except psutil.NoSuchProcess:
        print(f"PID {args.pid} does not exist.")
        return

    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "pid", "process_name", "rss_bytes", "rss_mib"])

        print(
            f"Sampling RSS for PID={args.pid} ({proc_name}) every {args.interval}s.\n"
            f"Writing CSV: {args.csv}\nPress Ctrl+C to stop and generate {args.png}."
        )

        try:
            while True:
                # Process may exit between iterations
                try:
                    rss_bytes = proc.memory_info().rss
                except psutil.NoSuchProcess:
                    print(f"PID {args.pid} exited; stopping sampler.")
                    break

                ts = datetime.now(timezone.utc).isoformat()
                writer.writerow([
                    ts,
                    args.pid,
                    proc_name,
                    rss_bytes,
                    round(bytes_to_mib(rss_bytes), 3),
                ])
                f.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopping sampler...")

    generate_plot(args.csv, args.png)


if __name__ == "__main__":
    main()
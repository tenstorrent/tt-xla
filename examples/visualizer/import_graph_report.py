# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Import captured ttnn graph reports into ttnn-visualizer SQLite databases.

Offline step: needs the ttnn Python package, not a device.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="capture JSON files, or directories whose captures are merged into one database",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("visualizer_dbs"),
        help="directory to create the databases in",
    )
    parser.add_argument(
        "--svgs", action="store_true", help="also render per-report SVGs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from ttnn.graph_report import import_report

    for report in args.reports:
        if not report.exists():
            raise SystemExit(f"no such report: {report}")
        out_dir = args.out / report.stem
        db = import_report(report, out_dir, generate_svgs=args.svgs)
        print(f"{report} -> {db}")

    print(f"\nOpen {args.out} in ttnn-visualizer as a local report directory.")


if __name__ == "__main__":
    main()

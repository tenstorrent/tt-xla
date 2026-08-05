# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Capture ttnn graph reports from a torch_xla run on TT hardware.

The capture is opened by tt-mlir's runtime inside ProgramExecutor::execute(), not
from Python: torch_xla dispatches execution to a worker thread and ttnn's
GraphTracker keeps its processors in thread_local storage, so a capture opened
around the model call sees nothing.

Import the resulting JSONs with import_graph_report.py.
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("graph_reports"),
        help="directory to write capture JSONs to",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="program executions to run before capturing; skips warm-up compilation",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="program executions per capture window, all merged into one report",
    )
    parser.add_argument("--steps", type=int, default=3, help="model invocations to run")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The runtime reads these once, on the first program execution.
    os.environ["TT_RUNTIME_GRAPH_CAPTURE_DIR"] = str(args.out.resolve())
    os.environ["TT_RUNTIME_GRAPH_CAPTURE_SKIP"] = str(args.skip)
    os.environ["TT_RUNTIME_GRAPH_CAPTURE_COUNT"] = str(args.count)

    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, 3, 1)
            self.fc = nn.Linear(32 * 26 * 26, 10)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            return self.fc(x.flatten(1))

    xr.set_device_type("TT")
    torch.manual_seed(42)

    device = xm.xla_device()
    model = ConvNet().to(dtype=torch.bfloat16).eval()
    model.compile(backend="tt")
    model = model.to(device)

    with torch.no_grad():
        for step in range(args.steps):
            inp = torch.full((4, 1, 28, 28), step + 1.0, dtype=torch.bfloat16)
            out = model(inp.to(device))
            print(f"step {step}: {out.cpu()[0, :3].tolist()}")

    reports = sorted(p.name for p in args.out.glob("*.json"))
    if not reports:
        raise SystemExit(
            f"no reports in {args.out} — the tt-mlir runtime in this build has no "
            "graph-capture hook (see README.md)"
        )
    print(f"\n{len(reports)} report(s) in {args.out}:")
    for name in reports:
        print(f"  {name}")


if __name__ == "__main__":
    main()

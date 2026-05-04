# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates that a single codegen_py() call produces multiple graphs
### when the model's forward contains a graph break.
###
### forward(a, b, c) computes s = a + b, calls dynamo.graph_break() so the
### trace splits, and then returns c * s. The two halves are compiled
### independently, so codegen runs twice — graph_0 holds the add and
### graph_1 holds the multiply.

import shutil
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

EXPORT_PATH = "graph_break_codegen"


class GraphBreakModel(nn.Module):
    def forward(self, a, b, c):
        s = a + b
        # Force a graph break
        dynamo.graph_break()
        return c * s


def main():
    xr.set_device_type("TT")

    model = GraphBreakModel()
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    c = torch.randn(32, 32)

    codegen_py(model, a, b, c, export_path=EXPORT_PATH)


def test_graph_break_codegen():
    """One codegen_py call should produce two graphs around the break."""
    output_dir = Path(EXPORT_PATH)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert (
            output_dir / "graph_0"
        ).is_dir(), f"Expected first graph at {output_dir}/graph_0"
        assert (
            output_dir / "graph_1"
        ).is_dir(), f"Expected second graph at {output_dir}/graph_1"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()

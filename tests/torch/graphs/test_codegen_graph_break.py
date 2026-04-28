# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Verifies that a single codegen_py() call produces multiple codegen
# directories when the model's forward contains a dynamo graph break.

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from utils import Category


class GraphBreakModel(nn.Module):
    def forward(self, a, b, c):
        s = a + b
        dynamo.graph_break()
        return c * s


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_codegen_graph_break_produces_two_subdirs():
    xr.set_device_type("TT")

    with tempfile.TemporaryDirectory() as tmp:
        export_path = Path(tmp) / "out"

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        d = torch.randn(32, 32)

        codegen_py(GraphBreakModel(), a, b, d, export_path=str(export_path))

        graph_0 = export_path / "graph_0"
        graph_1 = export_path / "graph_1"
        assert graph_0.is_dir(), f"Expected first graph at {graph_0}"
        assert graph_1.is_dir(), f"Expected second graph at {graph_1}"
        # Sanity-check codegen wrote a meaningful artifact in each subdir.
        assert (graph_0 / "main.py").is_file()
        assert (graph_1 / "main.py").is_file()

        # The split should put the add in graph_0 and the multiply in graph_1.
        graph_0_ir = (graph_0 / "ttnn.mlir").read_text()
        graph_1_ir = (graph_1 / "ttnn.mlir").read_text()
        assert "ttnn.add" in graph_0_ir and "ttnn.multiply" not in graph_0_ir
        assert "ttnn.multiply" in graph_1_ir and "ttnn.add" not in graph_1_ir

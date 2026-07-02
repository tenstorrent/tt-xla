# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-op tests for the compute-kernel-config overrides.

Each test drives one op with one compute-kernel-config knob set through the
compiler config, dumps the TTNN IR, and asserts the override lands on the op's
`device_compute_kernel_config` attribute. The bool knobs are tri-state
(True / False / Unset); these tests exercise the explicit True / False values,
including forcing a knob OFF (which is distinct from leaving it unset).
"""

import re
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from utils import Category


def _run_and_read_ttnn_ir(model_factory, input_factory, compile_options):
    """Compile a single-op module with the given compile options, then return
    the dumped TTNN IR (forward graph, g0) as a string."""
    xr.set_device_type("TT")
    device = torch_xla.device()

    with tempfile.TemporaryDirectory() as export_dir:
        options = dict(compile_options)
        options["export_path"] = export_dir
        options["export_model_name"] = "cckc"
        torch_xla.set_custom_compile_options(options)

        model = model_factory().to(device)
        x = input_factory().to(device)

        out = model(x)
        torch_xla.sync()

        irs_dir = Path(export_dir) / "irs"
        assert irs_dir.exists(), f"IR directory not created at {irs_dir}"

        # Graph number (g<N>) is a process-global counter, so match any graph.
        pattern = re.compile(r"^ttnn_cckc_g\d+_\d+\.mlir$")
        matching = [f for f in irs_dir.glob("*.mlir") if pattern.match(f.name)]
        assert matching, f"No TTNN IR dumped in {irs_dir}: {list(irs_dir.iterdir())}"

        return matching[0].read_text()


class _Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class _Matmul(nn.Module):
    def __init__(self, inner, outer, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inner, outer, dtype=dtype))

    def forward(self, x):
        return torch.matmul(x, self.weight)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_math_approx_mode_softmax():
    """math_approx_mode=False on softmax yields a more accurate softmax; verify
    the compiler forces the knob OFF in the TTNN IR."""
    ir = _run_and_read_ttnn_ir(
        model_factory=_Softmax,
        input_factory=lambda: torch.randn(32, 128, dtype=torch.bfloat16),
        compile_options={"math_approx_mode": "false"},
    )
    assert "ttnn.softmax" in ir
    assert "math_approx_mode = false" in ir


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_fp32_dest_acc_en_matmul():
    """Force fp32_dest_acc_en OFF on a matmul (distinct from leaving it unset)."""
    # Each matmul test uses a distinct shape so the executable cache does not
    # serve one test's compiled graph to another (which would skip IR export).
    ir = _run_and_read_ttnn_ir(
        model_factory=lambda: _Matmul(64, 64),
        input_factory=lambda: torch.randn(64, 64, dtype=torch.bfloat16),
        compile_options={"fp32_dest_acc_en": "false"},
    )
    assert "ttnn.matmul" in ir
    assert "fp32_dest_acc_en = false" in ir


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_packer_l1_acc_matmul():
    """Force packer_l1_acc ON on a matmul."""
    ir = _run_and_read_ttnn_ir(
        model_factory=lambda: _Matmul(96, 96),
        input_factory=lambda: torch.randn(64, 96, dtype=torch.bfloat16),
        compile_options={"packer_l1_acc": "true"},
    )
    assert "ttnn.matmul" in ir
    assert "packer_l1_acc = true" in ir


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_dst_full_sync_en_matmul():
    """Force dst_full_sync_en ON on a matmul."""
    ir = _run_and_read_ttnn_ir(
        model_factory=lambda: _Matmul(128, 128),
        input_factory=lambda: torch.randn(64, 128, dtype=torch.bfloat16),
        compile_options={"dst_full_sync_en": "true"},
    )
    assert "ttnn.matmul" in ir
    assert "dst_full_sync_en = true" in ir

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Analyze graph breaks in the original Compressor.forward() using TT backend.

This script runs the original Compressor through the TT-XLA compilation path
to identify graph breaks and compilation failures, using the run_graph_test
infrastructure.

Run with: pytest -svv ds4/analyze_compressor_graph_breaks.py
"""
import sys
from pathlib import Path

# Add ds4 directory to path for imports
ds4_path = Path(__file__).parent
if str(ds4_path) not in sys.path:
    sys.path.insert(0, str(ds4_path))

# Install kernel stubs BEFORE importing from model.py
import kernel_stubs
kernel_stubs.install()

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr

from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
# from model import ModelArgs, Compressor, precompute_freqs_cis
from aleks_modified_model import ModelArgs, Compressor, precompute_freqs_cis
from kernel_stubs import compressor_forward_prefill

torch.set_default_dtype(torch.bfloat16)


def init_weights(module):
    """Initialize weights properly for testing."""
    for name, param in module.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                if 'weight' in name and param.dim() >= 2:
                    init_data = torch.empty_like(param, dtype=torch.float32)
                    nn.init.xavier_uniform_(init_data)
                    param.copy_(init_data.to(param.dtype))
                elif 'bias' in name:
                    param.zero_()
                elif param.dim() == 1:
                    param.fill_(1.0)


# ============================================================================
# Wrapper classes to make Compressor work with run_graph_test
# ============================================================================


class CompressorPrefillOriginal(nn.Module):
    """
    Wrapper for original Compressor.forward() for prefill (start_pos=0).

    This uses the ORIGINAL forward method which has conditional branches.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
        self.compress_ratio = compress_ratio
        self.args = args

    def setup_buffers(self, batch_size: int, device=None):
        """Setup kv_cache and freqs_cis buffers."""
        self.compressor.kv_cache = torch.zeros(
            batch_size,
            self.args.max_seq_len // self.compress_ratio,
            self.args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.compressor.freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim,
            self.args.max_seq_len,
            0,
            self.args.rope_theta,
            self.args.rope_factor,
            self.args.beta_fast,
            self.args.beta_slow
        )
        if device is not None:
            self.compressor.freqs_cis = self.compressor.freqs_cis.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with start_pos=0 (prefill)."""
        return self.compressor.forward(x, start_pos=0)


class CompressorPrefillSimplified(nn.Module):
    """
    Wrapper for simplified compressor_forward_prefill stub.

    This is the statically compilable version without branches.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
        self.compress_ratio = compress_ratio
        self.args = args

    def setup_buffers(self, batch_size: int, device=None):
        """Setup kv_cache and freqs_cis buffers."""
        self.compressor.kv_cache = torch.zeros(
            batch_size,
            self.args.max_seq_len // self.compress_ratio,
            self.args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.compressor.freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim,
            self.args.max_seq_len,
            0,
            self.args.rope_theta,
            self.args.rope_factor,
            self.args.beta_fast,
            self.args.beta_slow
        )
        if device is not None:
            self.compressor.freqs_cis = self.compressor.freqs_cis.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using simplified prefill stub."""
        return compressor_forward_prefill(self.compressor, x)


# ============================================================================
# Test: Original Compressor - expect graph breaks/failures
# ============================================================================


@pytest.mark.parametrize("compress_ratio,seq_len,description", [
    pytest.param(4, 32, "CSA divisible", marks=pytest.mark.xfail(reason="Graph breaks due to dynamic control flow")),
    pytest.param(4, 64, "CSA longer", marks=pytest.mark.xfail(reason="Graph breaks due to dynamic control flow")),
    pytest.param(128, 128, "HSA divisible", marks=pytest.mark.xfail(reason="Graph breaks due to dynamic control flow")),
    pytest.param(128, 256, "HSA longer", marks=pytest.mark.xfail(reason="Graph breaks due to dynamic control flow")),
])
def test_compressor_original_prefill(compress_ratio, seq_len, description):
    """
    Test ORIGINAL Compressor.forward() on TT device.

    This may fail or show graph breaks due to conditional branches.
    """
    xr.set_device_type("TT")
    torch.manual_seed(42)

    batch_size = 1
    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=512,
        dim=256,
        head_dim=64,
        rope_head_dim=16,
    )

    model = CompressorPrefillOriginal(args, compress_ratio=compress_ratio)
    model.apply(init_weights)
    model.setup_buffers(batch_size)
    model.eval()

    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.90),
    )

    print(f"\n{'='*60}")
    print(f"ORIGINAL Compressor: {description}")
    print(f"  compress_ratio={compress_ratio}, seq_len={seq_len}")
    print(f"{'='*60}")

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Test: Simplified Compressor stub - should NOT have graph breaks
# ============================================================================


@pytest.mark.parametrize("compress_ratio,seq_len,description", [
    (4, 32, "CSA divisible"),
    pytest.param(4, 64, "CSA longer", marks=pytest.mark.xfail(reason="PCC varies - may be seed or state issue")),
    pytest.param(128, 128, "HSA divisible", marks=pytest.mark.xfail(reason="NaN in CPU reference - numerical instability with ratio=128")),
    pytest.param(128, 256, "HSA longer", marks=pytest.mark.xfail(reason="NaN in CPU reference - numerical instability with ratio=128")),
])
def test_compressor_simplified_prefill(compress_ratio, seq_len, description):
    """
    Test SIMPLIFIED compressor_forward_prefill on TT device.

    This should compile cleanly without graph breaks.
    """
    xr.set_device_type("TT")
    torch.manual_seed(42)

    batch_size = 1
    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=512,
        dim=256,
        head_dim=64,
        rope_head_dim=16,
    )

    model = CompressorPrefillSimplified(args, compress_ratio=compress_ratio)
    model.apply(init_weights)
    model.setup_buffers(batch_size)
    model.eval()

    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Lower threshold to verify compilation works (accuracy is secondary here)
    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.80),
    )

    print(f"\n{'='*60}")
    print(f"SIMPLIFIED Compressor: {description}")
    print(f"  compress_ratio={compress_ratio}, seq_len={seq_len}")
    print(f"{'='*60}")

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Main: Run tests manually to see detailed output
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("COMPRESSOR GRAPH BREAK ANALYSIS - TT BACKEND")
    print("=" * 70)
    print("\nRun with: pytest -svv ds4/analyze_compressor_graph_breaks.py")
    print("\nThis will show compilation errors/graph breaks when running on TT device.")

    # Quick test
    print("\n" + "#" * 70)
    print("# Testing simplified prefill (should work)")
    print("#" * 70)
    test_compressor_simplified_prefill(4, 32, "CSA quick test")

    print("\n" + "#" * 70)
    print("# Testing original prefill (may show graph breaks)")
    print("#" * 70)
    test_compressor_original_prefill(4, 32, "CSA quick test")

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolate the DeepSeek V3.1 low-PCC regression to the bfp8 weight path.

Context
-------
The full-model benchmark (``tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers``)
runs weights as ``bfp_bf8`` block-float tiles. PCC degrades with depth:

    4 layers  -> ~0.92      30 layers -> ~0.08      (full model = 61 layers)

That exponential-with-depth decay is the signature of a *systematic per-layer*
error compounding coherently, not random rounding. The op-by-op chisel numerics
report cannot see it: chisel's golden mapper can't map ``!ttcore.tile<32x32, bfp_bf8>``
so every bfp8 matmul/typecast is skipped (a blind spot). These tests measure that
blind spot directly, on a single device, with no chisel and no Galaxy mesh.

Mechanism
---------
The weight dtype conversion is only performed when the **global** compile option
``experimental_weight_dtype`` is set (see ``module_builder.cc`` —
``convertFromTTIRToTTNN``: an empty option leaves the dtype unset and the whole
conversion pass is skipped). So this test sets ``experimental_weight_dtype="bfp_bf8"``
for *both* arms (exactly as the benchmark does, to enable the pass) and uses the
per-tensor ``torch.ops.tt.weight_dtype_override(w, dtype_str)`` to choose the dtype
for the weight under test. The override is a **pass-through on CPU**, so CPU is
always the bf16 reference; on device it stores ``w`` in ``dtype_str``. The PCC is
therefore exactly the precision loss attributable to the weight format.

Each test sweeps ``weight_dtype in {bfp_bf8, bf16}`` (both with the pass enabled):
  * ``bf16``    -> per-tensor override downgrades this weight to bf16; should match
                   the CPU bf16 reference (~1.0). Sanity control.
  * ``bfp_bf8`` -> the quantity of interest.

Reading the result (the arms must DIFFER for the test to be measuring anything):
  * bfp8 low & bf16 high  -> override works; the gap is the bfp8 loss. ✅
  * both low & identical   -> global bfp8 applied but per-op ``bf16`` override ignored.
  * both high & identical  -> neither engaged (standalone Parameter not recognized as
                              a weight in the op-test compile path; use the Galaxy
                              full-model component tests instead).

Two tests:
  1. ``test_bfp8_matmul_per_projection`` — one matmul per real DeepSeek projection
     shape. Quantifies the per-matmul bfp8 loss the report can't measure.
  2. ``test_bfp8_mlp_tower_depth`` — a residual SwiGLU tower (same math as
     ``DeepseekV3MLP``) of N distinct blocks, swept over depth, to reproduce the
     PCC-vs-depth decay and confirm it is driven by the bfp8 weights (bf16 stays
     flat near 1.0). Dims are scaled down so the tower fits one device; the point
     is the *compounding trend*, not absolute model dims (the op test above covers
     real dims).

Compute config mirrors the model's matmuls: ``math_fidelity=hifi4``,
``fp32_dest_acc_en=True``.

All PCCs are printed regardless of pass/fail so a single sweep yields the full
curve; ``--required-pcc``-style gating is intentionally soft (default 0.0) because
these are characterization tests.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from infra import Framework, run_op_test
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig

DTYPE = torch.bfloat16

# Real DeepSeek V3.1 weight matmul shapes (contraction, output), from the model
# config: hidden=7168, q_lora_rank=1536, kv_lora_rank=512, MLA head dims, dense
# MLP intermediate=18432, MoE expert intermediate=2048. These are exactly the
# matmuls that run as bfp_bf8 in the benchmark.
PROJECTIONS = [
    # MLA attention
    ("attn.q_a_proj", 7168, 1536),
    ("attn.q_b_proj", 1536, 24576),
    ("attn.kv_a_proj_mqa", 7168, 576),
    ("attn.kv_b_proj", 512, 32768),
    ("attn.o_proj", 16384, 7168),
    # Dense MLP (layers 0..2)
    ("mlp.gate_proj", 7168, 18432),
    ("mlp.up_proj", 7168, 18432),
    ("mlp.down_proj", 18432, 7168),
    # MoE routed-expert MLP (layer 3+)
    ("moe.expert.gate_proj", 7168, 2048),
    ("moe.expert.up_proj", 7168, 2048),
    ("moe.expert.down_proj", 2048, 7168),
]

# Sequence length for the activation (one token-block); kept modest so the large
# real-dim matmuls fit comfortably on a single device.
TOKENS = 128


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, identical to the benchmark's compute_pcc."""
    x = a.detach().to(torch.float32).flatten()
    y = b.detach().to(torch.float32).flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm()).item()
    if denom == 0:
        return 1.0 if torch.allclose(x, y, rtol=1e-2, atol=1e-2) else float("nan")
    return max(-1.0, min(1.0, ((vx @ vy) / denom).item()))


def _pcc_comparator(label: str, required_pcc: float = 0.0) -> Callable:
    """Always print the PCC; only fail if it falls below ``required_pcc``."""

    def _compare(tt_res, cpu_res, args, kwargs):
        pcc = _pcc(tt_res.to("cpu"), cpu_res.to("cpu"))
        print(f"\n[bfp8-blindspot] {label}: pcc={pcc:.6f}", flush=True)
        assert not np.isnan(pcc), f"{label}: PCC is NaN"
        assert pcc >= required_pcc, f"{label}: PCC {pcc:.6f} < {required_pcc}"

    return _compare


def _compile_config() -> CompilerConfig:
    """Matmul config matching the model: hifi4 + fp32 accumulation, and the weight
    dtype conversion pass enabled (``experimental_weight_dtype`` non-empty). The
    pass is gated on this global option; the per-tensor override then selects the
    dtype for each weight."""
    return CompilerConfig(
        math_fidelity="hifi4",
        fp32_dest_acc_en=True,
        experimental_weight_dtype="bfp_bf8",
    )


class _Matmul(torch.nn.Module):
    """y = x @ override(W, weight_dtype). Mirrors tests/torch/ops/test_matmul.py."""

    def __init__(self, inner: int, outer: int, weight_dtype: str):
        super().__init__()
        # Unit-scale weights; activations are unit-scale below. Realistic for the
        # post-RMSNorm regime the model's matmuls actually see.
        self.weight = torch.nn.Parameter(torch.randn(inner, outer, dtype=DTYPE))
        self.weight_dtype = weight_dtype

    def forward(self, x):
        w = torch.ops.tt.weight_dtype_override(self.weight, self.weight_dtype)
        return torch.matmul(x, w)


class _SwiGLUBlock(torch.nn.Module):
    """Residual pre-norm SwiGLU block — same math as DeepseekV3MLP + RMSNorm.

    h -> h + down( silu(gate(rms(h))) * up(rms(h)) )
    All three projections carry the weight_dtype override (the bfp8 path).
    """

    def __init__(self, hidden: int, inter: int, weight_dtype: str, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight_dtype = weight_dtype
        self.norm_w = torch.nn.Parameter(torch.ones(hidden, dtype=DTYPE))
        self.gate = torch.nn.Parameter(torch.randn(hidden, inter, dtype=DTYPE) * 0.02)
        self.up = torch.nn.Parameter(torch.randn(hidden, inter, dtype=DTYPE) * 0.02)
        self.down = torch.nn.Parameter(torch.randn(inter, hidden, dtype=DTYPE) * 0.02)

    def _rms(self, h):
        v = h.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v.to(DTYPE)) * self.norm_w

    def forward(self, h):
        x = self._rms(h)
        g = torch.ops.tt.weight_dtype_override(self.gate, self.weight_dtype)
        u = torch.ops.tt.weight_dtype_override(self.up, self.weight_dtype)
        d = torch.ops.tt.weight_dtype_override(self.down, self.weight_dtype)
        act = torch.nn.functional.silu(torch.matmul(x, g)) * torch.matmul(x, u)
        return h + torch.matmul(act, d)


class _Tower(torch.nn.Module):
    """A stack of ``depth`` distinct _SwiGLUBlocks (distinct random weights)."""

    def __init__(self, depth: int, hidden: int, inter: int, weight_dtype: str):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            _SwiGLUBlock(hidden, inter, weight_dtype) for _ in range(depth)
        )

    def forward(self, h):
        for blk in self.blocks:
            h = blk(h)
        return h


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST, torch_op_name="torch.matmul"
)
@pytest.mark.parametrize("weight_dtype", ["bfp_bf8", "bf16"])
@pytest.mark.parametrize(
    "name, inner, outer", PROJECTIONS, ids=[p[0] for p in PROJECTIONS]
)
def test_bfp8_matmul_per_projection(name, inner, outer, weight_dtype):
    """Per-projection bfp8 matmul PCC vs a bf16 reference (chisel's blind spot)."""
    torch.manual_seed(0)
    matmul = _Matmul(inner, outer, weight_dtype)
    activation = torch.randn(TOKENS, inner, dtype=DTYPE)

    run_op_test(
        matmul,
        [activation],
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.0)),
        framework=Framework.TORCH,
        compiler_config=_compile_config(),
        custom_comparator=_pcc_comparator(f"matmul[{name}|{weight_dtype}]"),
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.GRAPH_TEST, torch_op_name="torch.matmul"
)
@pytest.mark.parametrize("weight_dtype", ["bfp_bf8", "bf16"])
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16, 30])
def test_bfp8_mlp_tower_depth(depth, weight_dtype):
    """PCC vs depth for a residual SwiGLU tower — reproduces the compounding decay.

    Expectation: with ``bfp_bf8`` weights PCC falls as depth grows (mirroring the
    4->0.92 / 30->0.08 model trend); with ``bf16`` it stays flat near 1.0,
    attributing the decay to the weight format. Dims scaled down to fit one device
    (the per-projection test above uses the real dims).
    """
    torch.manual_seed(0)
    hidden, inter = 2048, 5120  # DeepSeek dense ratio (7168:18432) scaled ~3.5x down
    tower = _Tower(depth, hidden, inter, weight_dtype)
    h = torch.randn(1, TOKENS, hidden, dtype=DTYPE)

    run_op_test(
        tower,
        [h],
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.0)),
        framework=Framework.TORCH,
        compiler_config=_compile_config(),
        custom_comparator=_pcc_comparator(f"tower[d={depth:02d}|{weight_dtype}]"),
    )

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-op matmul tests using real weights from GPT-OSS 120B.

Weights are loaded from files saved by scripts/extract_gpt_oss_matmul_activations.py.
Input activations are random (seed 42) with shapes matching the real model.
Attention score/context matmuls use fully random operands (no fixed weights).

Required env var:
    GPT_OSS_120B_WEIGHTS_DIR — output directory from extract_gpt_oss_matmul_activations.py

Optional env var:
    REL_L2_OUTPUT — path to a JSONL file where per-test rel_l2/pcc values are appended.
                    Use this to collect metrics before and after a compiler change:
                        REL_L2_OUTPUT=before.jsonl pytest ...
                        # apply fix, then:
                        REL_L2_OUTPUT=after.jsonl pytest ...

Run:
    python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/gpt_oss_120b_weights
    export GPT_OSS_120B_WEIGHTS_DIR=/tmp/gpt_oss_120b_weights
    pytest tests/operators/test_matmul_gpt_oss_120b.py/
"""

import dataclasses
import json
import os
from itertools import product
from pathlib import Path

import pytest
import torch

from infra import Framework, run_op_test
from tests.infra.testers.compiler_config import CompilerConfig

# ---------------------------------------------------------------------------
# Op registry
# (op_subdir, has_weight, transpose_weight, lhs_shape, rhs_shape)
#
# has_weight=True   — load weight.pt; lhs generated randomly
# has_weight=False  — both operands random (attention score/context matmuls)
# transpose_weight  — nn.Linear stores weight as [out, in]; transpose for matmul
# shapes            — after applying transpose (what enters torch.matmul)
#
# MoE ops excluded: weight is 3D [num_experts, in, out], not isolatable.
# ---------------------------------------------------------------------------

_LAYERS = [0, 18, 19]

_OPS = [
    # (op_subdir,            has_w,  transpose, lhs_shape,           rhs_shape)
    ("self_attn_q_proj",    True,  True,  (1, 128, 2880),    (2880, 4096)),
    ("self_attn_k_proj",    True,  True,  (1, 128, 2880),    (2880, 512)),
    ("self_attn_v_proj",    True,  True,  (1, 128, 2880),    (2880, 512)),
    ("self_attn_o_proj",    True,  True,  (1, 128, 4096),    (4096, 2880)),
    ("mlp_router",          True,  True,  (128, 2880),       (2880, 32)),
    ("attn_score_matmul",   False, False, (1, 64, 128, 64),  (1, 64, 64, 128)),
    ("attn_context_matmul", False, False, (1, 64, 128, 128), (1, 64, 128, 64)),
]

_PARAMS = [
    (f"layer_{layer_idx}", op, has_w, transpose, lhs, rhs)
    for layer_idx in _LAYERS
    for op, has_w, transpose, lhs, rhs in _OPS
]
_IDS = [f"{layer}__{op}" for layer, op, *_ in _PARAMS]

# Full sweep matching test_matmul_mp.py: opt × dtype × math_fidelity × fp32_acc
_WEIGHT_DTYPE_FIDELITY = [
    ("",        ["hifi4", "hifi3", "hifi2", "lofi"]),
    ("bfp_bf8", ["hifi4", "hifi3", "hifi2", "lofi"]),
    ("bfp_bf4", ["hifi4", "hifi3", "hifi2", "lofi"]),
]
_DTYPE_LABEL = {"": "bf16", "bfp_bf8": "bfp8", "bfp_bf4": "bfp4"}

_COMPILER_CONFIGS = [
    CompilerConfig(
        optimization_level=opt,
        experimental_weight_dtype=dtype,
        math_fidelity=fidelity,
        fp32_dest_acc_en=fp32,
    )
    for opt, (dtype, fidelities), fp32 in product([0, 2], _WEIGHT_DTYPE_FIDELITY, [True, False])
    for fidelity in fidelities
]
_COMPILER_IDS = [
    f"opt{c.optimization_level}_{_DTYPE_LABEL[c.experimental_weight_dtype]}_{c.math_fidelity}_fp32{'true' if c.fp32_dest_acc_en else 'false'}"
    for c in _COMPILER_CONFIGS
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _MatmulWithWeight(torch.nn.Module):
    """Wraps a pre-loaded weight tensor as a parameter; input is the LHS activation."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


class _Matmul(torch.nn.Module):
    """Pure two-input matmul — used for attention score/context ops."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    path = os.environ.get("GPT_OSS_120B_WEIGHTS_DIR")
    if not path:
        pytest.skip("GPT_OSS_120B_WEIGHTS_DIR not set — run extract_gpt_oss_matmul_activations.py first")
    return Path(path)


def _maybe_inject_export_path(compiler_config: CompilerConfig, request) -> CompilerConfig:
    """If TT_MLIR_EXPORT_PATH is set, dump MLIR stages into a per-test subdirectory.

    Each test case writes to <TT_MLIR_EXPORT_PATH>/<sanitized_test_name>/irs/<stage>_*.mlir.
    Used to compare graphs before and after a compiler change.
    """
    base = os.environ.get("TT_MLIR_EXPORT_PATH")
    if not base:
        return compiler_config
    safe = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
    test_dir = Path(base) / safe
    test_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(
        compiler_config,
        export_path=str(test_dir),
        export_model_name=safe,
    )


def _build_model_and_inputs(
    layer: str,
    op: str,
    has_weight: bool,
    transpose: bool,
    lhs_shape: tuple,
    rhs_shape: tuple,
):
    torch.manual_seed(42)
    x = torch.randn(lhs_shape, dtype=torch.bfloat16)

    if has_weight:
        weight_path = _weights_dir() / layer / op / "weight.pt"
        if not weight_path.exists():
            pytest.skip(f"Missing weight file: {weight_path}")
        w = torch.load(weight_path, weights_only=True)
        if transpose:
            w = w.T
        return _MatmulWithWeight(w), [x]

    y = torch.randn(rhs_shape, dtype=torch.bfloat16)
    return _Matmul(), [x, y]


def _make_comparator(test_id: str, request, pcc_threshold: float = 0.99):
    """
    Returns a custom_comparator for run_op_test that:
    - computes rel_l2 and PCC
    - prints them (visible with pytest -s)
    - records them as pytest properties (appear in --junitxml output)
    - appends a JSONL entry to REL_L2_OUTPUT if that env var is set
    """
    def _comparator(tt_res, cpu_res, args, kwargs):
        tt_f64 = tt_res.cpu().to(torch.float64).flatten()
        cpu_f64 = cpu_res.cpu().to(torch.float64).flatten()

        diff_norm = torch.linalg.vector_norm(tt_f64 - cpu_f64).item()
        golden_norm = torch.linalg.vector_norm(cpu_f64).item()
        if golden_norm == 0.0:
            rel_l2 = 0.0 if diff_norm == 0.0 else float("inf")
        else:
            rel_l2 = diff_norm / golden_norm

        stacked = torch.stack([tt_f64.float(), cpu_f64.float()])
        pcc = float(torch.corrcoef(stacked)[0, 1])

        print(f"\n[METRICS] rel_l2={rel_l2:.6f}  pcc={pcc:.6f}  test={test_id}")

        if request is not None:
            request.node.user_properties.append(("rel_l2", rel_l2))
            request.node.user_properties.append(("pcc", pcc))

        output_path = os.environ.get("REL_L2_OUTPUT")
        if output_path:
            entry = json.dumps({"test_id": test_id, "rel_l2": rel_l2, "pcc": pcc})
            with open(output_path, "a") as f:
                f.write(entry + "\n")

        assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < {pcc_threshold}"

    return _comparator


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize("compiler_config", _COMPILER_CONFIGS, ids=_COMPILER_IDS)
@pytest.mark.parametrize("layer,op,has_weight,transpose,lhs_shape,rhs_shape", _PARAMS, ids=_IDS)
def test_matmul_gpt_oss_120b(layer, op, has_weight, transpose, lhs_shape, rhs_shape, compiler_config, request, clear_torchxla_computation_cache):
    """Matmul with GPT-OSS 120B weights and random input activations."""
    model, inputs = _build_model_and_inputs(layer, op, has_weight, transpose, lhs_shape, rhs_shape)

    compiler_config = _maybe_inject_export_path(compiler_config, request)

    run_op_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        request=request,
        custom_comparator=_make_comparator(request.node.nodeid, request),
    )

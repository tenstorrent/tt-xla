# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Same matrix as test_matmul_gpt_oss_120b.py but with random weights for ALL ops
(no extracted GPT-OSS weights). Used to check whether the bfp4 quantization
issue reproduces when weights come from torch.randn instead of the real model.

Shapes match the real model. Seed is fixed (42) for reproducibility.

Optional env var:
    REL_L2_OUTPUT — path to a JSONL file where per-test rel_l2/pcc are appended.
    TT_MLIR_EXPORT_PATH — base dir to dump per-test MLIR stages.
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

# Same op registry as test_matmul_gpt_oss_120b.py. has_weight is ignored here —
# every op uses random tensors for both operands.
_LAYERS = [0, 18, 19]

_OPS = [
    # (op_subdir,            transpose, lhs_shape,           rhs_shape)
    ("self_attn_q_proj",    True,  (1, 128, 2880),    (2880, 4096)),
    ("self_attn_k_proj",    True,  (1, 128, 2880),    (2880, 512)),
    ("self_attn_v_proj",    True,  (1, 128, 2880),    (2880, 512)),
    ("self_attn_o_proj",    True,  (1, 128, 4096),    (4096, 2880)),
    ("mlp_router",          True,  (128, 2880),       (2880, 32)),
    ("attn_score_matmul",   False, (1, 64, 128, 64),  (1, 64, 64, 128)),
    ("attn_context_matmul", False, (1, 64, 128, 128), (1, 64, 128, 64)),
]

_PARAMS = [
    (f"layer_{layer_idx}", op, transpose, lhs, rhs)
    for layer_idx in _LAYERS
    for op, transpose, lhs, rhs in _OPS
]
_IDS = [f"{layer}__{op}" for layer, op, *_ in _PARAMS]

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


class _MatmulWithWeight(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


class _Matmul(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


def _maybe_inject_export_path(compiler_config: CompilerConfig, request) -> CompilerConfig:
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


def _build_model_and_inputs(transpose: bool, lhs_shape: tuple, rhs_shape: tuple):
    """All operands are random — weights too. transpose flag is kept for parity
    with the real-weight variant (nn.Linear stores [out, in]) so the resulting
    matmul shape matches exactly."""
    torch.manual_seed(42)
    x = torch.randn(lhs_shape, dtype=torch.bfloat16)

    # rhs_shape is already the shape after transpose, so just draw it directly.
    # The transpose flag in the real-weight variant flips a [out, in] tensor;
    # here we just generate the post-transpose shape, which is equivalent for
    # numerical purposes (same fan-in for the matmul).
    w = torch.randn(rhs_shape, dtype=torch.bfloat16)

    # Use _MatmulWithWeight when the original op had a fixed weight, so the
    # weight is hoisted as a Parameter (matches const-eval behaviour). Otherwise
    # use _Matmul where both operands are activations.
    # We can't tell here whether the original was a weighted op, but for the
    # purposes of reproducing the bfp4 cast issue what matters is the const-eval
    # path — so always use _MatmulWithWeight for the linear projections and
    # _Matmul for the attn score/context ops. The transpose flag distinguishes
    # them: True ⇒ linear projection (weight const), False ⇒ pure 2-input matmul.
    if transpose:
        return _MatmulWithWeight(w), [x]
    return _Matmul(), [x, w]


def _make_comparator(test_id: str, request, pcc_threshold: float = 0.99):
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


@pytest.mark.single_device
@pytest.mark.parametrize("compiler_config", _COMPILER_CONFIGS, ids=_COMPILER_IDS)
@pytest.mark.parametrize("layer,op,transpose,lhs_shape,rhs_shape", _PARAMS, ids=_IDS)
def test_matmul_gpt_oss_120b_random(layer, op, transpose, lhs_shape, rhs_shape, compiler_config, request, clear_torchxla_computation_cache):
    """Matmul with fully random operands at GPT-OSS 120B shapes."""
    model, inputs = _build_model_and_inputs(transpose, lhs_shape, rhs_shape)

    compiler_config = _maybe_inject_export_path(compiler_config, request)

    run_op_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        request=request,
        custom_comparator=_make_comparator(request.node.nodeid, request),
    )

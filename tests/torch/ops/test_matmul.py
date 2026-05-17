# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig

# Directory layout for the manual-TTNN-IR-edit workflow:
#   modules/irs/ttnn_<model_name>_<ts>.mlir   <- exported TTNN IR (compile run)
#   modules/fb_<model_name>_<ts>.ttnn         <- exported flatbuffer (compile run)
#   flatbuffers/<model_name>.ttnn             <- user-edited flatbuffer (load run)
TTNN_IR_EXPORT_DIR = Path(__file__).resolve().parents[3] / "modules"
TTNN_FB_LOAD_DIR = Path(__file__).resolve().parents[3] / "flatbuffers"


class Matmul(torch.nn.Module):
    def __init__(
        self, inner_dim, rhs_outer_dim, weight_dtype="bf16", dtype=torch.bfloat16
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(inner_dim, rhs_outer_dim, dtype=dtype)
        )
        self.weight_dtype = weight_dtype

    def forward(self, x):
        w = torch.ops.tt.weight_dtype_override(self.weight, self.weight_dtype)
        return torch.matmul(x, w)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("lhs_outer", [32, 64])
@pytest.mark.parametrize("rhs_outer", [32, 64])
@pytest.mark.parametrize("inner", [32, 64])
@pytest.mark.parametrize("weight_dtype", ["bfp_bf8", "bfp_bf4"])
def test_matmul_rhs_as_param(lhs_outer, rhs_outer, inner, weight_dtype):
    dtype = torch.bfloat16
    matmul = Matmul(inner, rhs_outer, weight_dtype=weight_dtype, dtype=dtype)
    compiler_config = CompilerConfig()
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=0.98)

    run_op_test_with_random_inputs(
        matmul,
        [(lhs_outer, inner)],
        dtype=dtype,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("math_fidelity", ["hifi2", "hifi4", "ttnn_default"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_matmul_mf_fp32_acc(math_fidelity, fp32_dest_acc_en):
    dtype = torch.bfloat16
    inner_dim = 64
    rhs_outer_dim = 64
    lhs_outer_dim = 64

    matmul = Matmul(inner_dim, rhs_outer_dim, dtype=dtype)

    model_name = f"matmul_mf_fp32_acc_{math_fidelity}_{fp32_dest_acc_en}"
    compiler_config = CompilerConfig(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        export_path=str(TTNN_IR_EXPORT_DIR),
        export_model_name=model_name,
    )

    # If a hand-edited flatbuffer for this config exists, reuse it (skips
    # `ttnnToFlatbuffer`); otherwise compile normally and export the IR/FB.
    fb_path = TTNN_FB_LOAD_DIR / f"{model_name}.ttnn"
    if fb_path.exists():
        compiler_config.flatbuffer_load_path = str(fb_path)

    run_op_test_with_random_inputs(
        matmul,
        [(lhs_outer_dim, inner_dim)],
        dtype=dtype,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "shard_spec",
    [("model", None), (None, "model")],
    ids=["shard_dim0", "shard_dim1"],
)
@pytest.mark.parametrize("weight_dtype", ["bfp_bf8", "bfp_bf4"])
def test_matmul_weight_dtype_override_multi_chip(weight_dtype, shard_spec):
    """
    Matmul with weight_dtype_override and weight sharded across devices.

    Tests two sharding axes:
    - shard_dim0: weight sharded on contraction dim — forces all-gather on weight path
    - shard_dim1: column-parallel, weight sharded on output dim

    Verifies that weight_dtype_override annotations survive through CCL operations.
    """
    dtype = torch.bfloat16
    inner_dim = 64
    rhs_outer_dim = 64
    lhs_outer_dim = 32

    matmul = Matmul(inner_dim, rhs_outer_dim, weight_dtype, dtype)

    def get_shard_spec(model, args, kwargs):
        return {model.weight: shard_spec}

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    activation = torch.randn(lhs_outer_dim, inner_dim, dtype=dtype)

    comparison_config = ComparisonConfig()
    if weight_dtype == "bfp_bf4":
        comparison_config.pcc = PccConfig(required_pcc=0.98)

    run_op_test(
        matmul,
        [activation],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

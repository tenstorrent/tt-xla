# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig


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
@pytest.mark.parametrize("accuracy_mode", ["accuracy", "performance"])
def test_matmul_accuracy_mode(accuracy_mode):
    dtype = torch.bfloat16
    inner_dim = 64
    rhs_outer_dim = 64
    lhs_outer_dim = 64

    matmul = Matmul(inner_dim, rhs_outer_dim, dtype=dtype)
    compiler_config = CompilerConfig(accuracy_mode=accuracy_mode)

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

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
@pytest.mark.parametrize("math_fidelity", ["hifi2", "hifi4", "ttnn_default"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_matmul_mf_fp32_acc(math_fidelity, fp32_dest_acc_en):
    dtype = torch.bfloat16
    inner_dim = 64
    rhs_outer_dim = 64
    lhs_outer_dim = 64

    matmul = Matmul(inner_dim, rhs_outer_dim, dtype=dtype)
    compiler_config = CompilerConfig(
        math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en
    )

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

class MatmulFromAnotherOp(torch.nn.Module):
    """
    Matmul whose operands are produced by another op (an elementwise add),
    mirroring the sweeps ``ModelFromAnotherOp`` for matmul:
    ``xx = add(x, x); yy = add(y, y); matmul(xx, yy)``.
    """

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return torch.matmul(xx, yy)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_matmul_mp_opt2_large_seq_len():
    """
    Single-chip mixed-precision matmul, port of the sweeps test vector:

        matmul_mp-FROM_ANOTHER_OP-
        {'compiler_config': 'mp_opt2_bfp8_fp32acctrue_hifi2'}-
        ((32, 128, 1024), (1024, 2048))-None-None

    Decoded compiler config ``mp_opt2_bfp8_fp32acctrue_hifi2``:
      - optimization_level = 2
      - weight_dtype       = bfp8  -> experimental_weight_dtype = "bfp_bf8"
      - fp32_accumulation  = True  -> fp32_dest_acc_en = True
      - math_fidelity      = hifi2
    """
    lhs_shape = (32, 128, 1024)
    rhs_shape = (1024, 2048)

    matmul = MatmulFromAnotherOp()

    compiler_config = CompilerConfig(
        optimization_level=2,
        experimental_weight_dtype="bfp_bf8",
        math_fidelity="hifi2",
        fp32_dest_acc_en=True,
    )

    # Match the sweeps value_checker: PCC 0.99, allclose rtol/atol 1e-2.
    comparison_config = ComparisonConfig()
    comparison_config.pcc = PccConfig(required_pcc=0.99)

    # Sweeps uses ValueRanges.SMALL == [-1, 1) with float32 inputs (dev_data_format=None).
    run_op_test_with_random_inputs(
        matmul,
        [lhs_shape, rhs_shape],
        minval=-1.0,
        maxval=1.0,
        dtype="float32",
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
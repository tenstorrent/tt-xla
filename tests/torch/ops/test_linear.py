# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from infra.evaluators import TorchComparisonEvaluator
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelLoader as GemmaModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelVariant as GemmaModelVariant,
)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("in_features", [32, 64])
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("experimental_weight_dtype", ["", "bfp_bf8", "bfp_bf4"])
def test_linear(
    batch_size,
    in_features,
    out_features,
    bias,
    experimental_weight_dtype,
):
    dtype = torch.bfloat16
    linear = Linear(in_features, out_features, bias=bias, dtype=dtype)
    compiler_config = CompilerConfig(
        experimental_weight_dtype=experimental_weight_dtype
    )
    comparison_config = ComparisonConfig()
    if experimental_weight_dtype == "bfp_bf4":
        comparison_config.pcc = PccConfig(required_pcc=0.98)

    run_op_test_with_random_inputs(
        linear,
        [(batch_size, in_features)],
        dtype=dtype,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_linear_torch_override():
    """
    Test that TorchFunctionOverride produces correct results for linear operations with
    4D input tensors by comparing override output against standard torch.nn.functional.linear.

    This tests the override in eager mode (without torch.compile) since the override
    has `not torch.compiler.is_compiling()` check.
    """
    from tt_torch.torch_overrides import torch_function_override

    dtype = torch.bfloat16
    in_features = 96
    out_features = 384

    input_tensor = torch.randn(1, 128, 128, in_features, dtype=dtype)
    weight = torch.randn(out_features, in_features, dtype=dtype)
    bias = torch.randn(out_features, dtype=dtype)

    # Temporarily disable the override to get golden output
    torch_function_override.__exit__(None, None, None)
    try:
        golden = F.linear(input_tensor, weight, bias)
    finally:
        torch_function_override.__enter__()  # Always re-enable it

    # Compute actual output with override active (eager mode, not compiled)
    output = F.linear(input_tensor, weight, bias)

    comparator = TorchComparisonEvaluator(ComparisonConfig())
    comparator.evaluate(output, golden)


class TensorParallelMLP(torch.nn.Module):
    """
    Simple two-layer MLP with per-tensor weight_dtype_override annotations.

    Mimics tensor-parallel LLM MLP pattern:
      - up_proj: column-parallel (weight sharded on output dim)
      - down_proj: row-parallel (weight sharded on input dim)
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        weight_dtype,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype
        )
        self.weight_dtype = weight_dtype

    def forward(self, x):
        up_w = torch.ops.tt.weight_dtype_override(
            self.up_proj.weight, self.weight_dtype
        )
        x = F.linear(x, up_w)
        x = F.relu(x)
        down_w = torch.ops.tt.weight_dtype_override(
            self.down_proj.weight, self.weight_dtype
        )
        x = F.linear(x, down_w)
        return x


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("weight_dtype", ["bfp_bf8", "bfp_bf4"])
def test_linear_tensor_parallel_per_tensor_weight_dtype(weight_dtype):
    """
    Tensor-parallel MLP with per-tensor weight dtype overrides.

    up_proj weight is column-parallel sharded ("model", None) — each device
    holds a slice of the output dimension. down_proj weight is row-parallel
    sharded (None, "model") — each device holds a slice of the input dimension.
    The matmul + CCL (all-reduce after down_proj) pattern tests that
    weight_dtype_override annotations survive through tensor-parallel CCL ops.
    """
    dtype = torch.bfloat16
    hidden_size = 64
    intermediate_size = 128

    mlp = TensorParallelMLP(hidden_size, intermediate_size, weight_dtype, dtype)

    def get_shard_spec(model, args, kwargs):
        return {
            model.up_proj.weight: ("model", None),
            model.down_proj.weight: (None, "model"),
        }

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    batch_size = 1
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    comparison_config = ComparisonConfig()
    if weight_dtype == "bfp_bf4":
        comparison_config.pcc = PccConfig(required_pcc=0.98)

    run_op_test(
        mlp,
        [hidden_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "mesh_shape,shard_spec",
    [
        ((1, 8), ("batch", "model")),
        ((2, 4), (None, "batch")),
    ],
)
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_gemma2_27b_lm_head(mesh_shape, shard_spec):
    loader = GemmaModelLoader(variant=GemmaModelVariant.GEMMA_2_27B_IT)
    config = loader.load_config()

    lm_head = Linear(
        config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16
    )

    def get_shard_spec(lm_head, args, kwargs):
        shard_specs = {}
        shard_specs[lm_head.linear.weight] = shard_spec
        return shard_specs

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    batch_size = 1
    seq_len = 1024
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size),
        dtype=torch.bfloat16,
    )

    comparison_config = ComparisonConfig()

    run_op_test(
        lm_head,
        [hidden_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

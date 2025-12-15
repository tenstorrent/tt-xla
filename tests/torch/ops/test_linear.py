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
from infra.comparators.torch_comparator import TorchComparator
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.infra.comparators.comparison_config import ComparisonConfig
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
@pytest.mark.parametrize("experimental_enable_weight_bfp8_conversion", [False, True])
def test_linear(
    batch_size,
    in_features,
    out_features,
    bias,
    experimental_enable_weight_bfp8_conversion,
):
    dtype = torch.bfloat16
    linear = Linear(in_features, out_features, bias=bias, dtype=dtype)
    compiler_config = CompilerConfig(
        experimental_enable_weight_bfp8_conversion=experimental_enable_weight_bfp8_conversion
    )

    run_op_test_with_random_inputs(
        linear,
        [(batch_size, in_features)],
        dtype=dtype,
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

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


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

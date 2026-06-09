# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


# duplicated to have all fusion tests in the same place, also it uses the F.rms_norm rather than pulling from HF model code
@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
def test_rmsnorm(request):
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, (256,), gamma)

    run_graph_test_with_random_inputs(
        rmsnorm,
        [(64, 256), (256,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
def test_rmsnorm_decomposed_aot(request):
    """
    Decomposed HF-style RMSNorm body traced through AOTAutograd. Without an
    aten-form pattern in RMSNormFusionProvider, this lowers as ~7 separate ops
    instead of a single ttir.rms_norm (regression repro for tt-xla#4507).
    """

    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        return gamma * x.to(input_dtype)

    run_graph_test_with_random_inputs(
        rmsnorm,
        [(64, 256), (256,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        torch_options={"tt_use_aot_autograd": True},
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("use_aot_autograd", [False, True])
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
def test_gemma_rmsnorm_fusion(request, use_aot_autograd):
    """
    Gemma-style RMSNorm: weight is upcast to fp32 and shifted by +1 before
    multiply, output cast back via .type_as. Exercises pattern_gemma /
    pattern_gemma_aot.
    """

    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + 1e-6)
        out = normed * (1.0 + gamma.float())
        return out.type_as(x)

    run_graph_test_with_random_inputs(
        rmsnorm,
        [(64, 256), (256,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        torch_options={"tt_use_aot_autograd": use_aot_autograd},
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("use_aot_autograd", [False, True])
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
def test_gemma4_rmsnorm_fusion(request, use_aot_autograd):
    """
    Gemma4-style RMSNorm: weight is upcast to fp32 (no +1), and torch.pow(x, -0.5)
    is used in place of torch.rsqrt. Exercises pattern_gemma4 / pattern_gemma4_aot.
    """

    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        mean_squared = x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6
        normed = x_fp32 * torch.pow(mean_squared, -0.5)
        out = normed * gamma.float()
        return out.type_as(x)

    run_graph_test_with_random_inputs(
        rmsnorm,
        [(64, 256), (256,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        torch_options={"tt_use_aot_autograd": use_aot_autograd},
        request=request,
    )

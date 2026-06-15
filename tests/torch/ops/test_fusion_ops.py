# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from infra.utilities.types import Framework
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tests.infra.evaluators import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


class _VLLMRMSNormShape(nn.Module):
    """Mirror of integrations/vllm_plugin/vllm_tt/layers/rmsnorm.py TTRMSNorm
    (no residual / no variance_size_override paths).

    Defined inline so this file does not pull in the vllm package at collection
    time. The differentiating feature is the operand order on the trailing
    multiply: x * weight rather than weight * x.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        return x * self.weight


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_llama_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype, request):

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = LlamaRMSNorm(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gpt_oss_20b_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype, request):

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = GptOssRMSNorm(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemma_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype, request):
    """Covers Gemma / Gemma2 / Gemma3, which all share the same RMSNorm shape
    (weight upcast and shifted by +1 before multiply, .type_as cast back)."""

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = GemmaRMSNorm(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemma4_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype, request):
    """Gemma4 default (with_scale=True): weight upcast (no +1 shift),
    torch.pow(mean_squared, -0.5) in place of torch.rsqrt."""

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = Gemma4RMSNorm(hidden_size, with_scale=True)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemma4_no_scale_rms_norm_fusion(
    batch_size, seq_len, hidden_size, dtype, request
):
    """Gemma4 with_scale=False: no learned weight (identity scale)."""

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = Gemma4RMSNorm(hidden_size, with_scale=False)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vllm_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype, request):
    """vLLM TTRMSNorm shape: cast happens before multiply with weight, and the
    multiply order is x * weight (instead of HF Llama's weight * x). Regression
    coverage for the vLLM-Llama RMSNorm fusion miss."""

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = _VLLMRMSNormShape(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )

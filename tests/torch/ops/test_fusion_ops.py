# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tests.infra.evaluators import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


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
@pytest.mark.parametrize("with_scale", [True, False])
def test_gemma4_rms_norm_fusion(
    batch_size, seq_len, hidden_size, dtype, with_scale, request
):
    """Gemma4: weight upcast (no +1 shift), torch.pow(mean_squared, -0.5) in
    place of torch.rsqrt. with_scale=False drops the trailing weighted-multiply
    (identity scale)."""

    options = {
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_enable_composite_ops": True,
    }

    model = Gemma4RMSNorm(hidden_size, with_scale=with_scale)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        request=request,
    )

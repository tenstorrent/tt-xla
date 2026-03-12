# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, ComparisonConfig, run_graph_test
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRotaryEmbedding,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GPTOSSModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelVariant as GPTOSSModelVariant,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelVariant as LlamaModelVariant,
)

SEQ_LEN = 1024

# ---------------------------------------------------------------------------
# Llama 3 8B tests
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["sdpa.ttnn.mlir"])
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
@pytest.mark.filecheck(["concatenate_heads.ttnn.mlir"])
def test_llama_3_8b_sdpa(request):
    config = LlamaModelLoader(variant=LlamaModelVariant.LLAMA_3_8B).load_config()
    config._attn_implementation = "eager"

    attention = LlamaAttention(config, layer_idx=0).to(torch.bfloat16)
    hidden_states = torch.randn(1, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = LlamaRotaryEmbedding(config=config)(hidden_states, position_ids)

    run_graph_test(
        attention,
        [hidden_states, (cos, sin), None, None],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


# ---------------------------------------------------------------------------
# GPT-OSS 20B tests
# ---------------------------------------------------------------------------


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "layer_idx", [0, 1]
)  # Much like GPT-3, GPT-Neo and some Gemma models, GPT-OSS uses alternating sliding window and full attention layers.
@pytest.mark.filecheck(["sdpa.ttnn.mlir"])
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
@pytest.mark.filecheck(["concatenate_heads.ttnn.mlir"])
def test_gpt_oss_20b_sdpa(layer_idx, request):
    config = GPTOSSModelLoader(variant=GPTOSSModelVariant.GPT_OSS_20B).load_config()
    config._attn_implementation = "eager"
    comparison_config = ComparisonConfig()
    comparison_config.pcc.disable()

    attention = GptOssAttention(config, layer_idx=layer_idx).to(torch.bfloat16)
    hidden_states = torch.randn(1, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = GptOssRotaryEmbedding(config=config)(hidden_states, position_ids)

    run_graph_test(
        attention,
        [hidden_states, (cos, sin), None, None],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        comparison_config=comparison_config,
        request=request,
    )

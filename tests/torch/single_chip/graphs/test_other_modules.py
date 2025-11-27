# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh

from utils import failed_runtime

from tests.utils import is_llmbox
from third_party.tt_forge_models.bert.masked_lm.pytorch.loader import (
    ModelLoader as BertModelLoader,
)
from third_party.tt_forge_models.bge_m3.pytorch.loader import (
    ModelLoader as BgeModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelLoader as GemmaModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelVariant as GemmaModelVariant,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.mistral.pytorch.loader import (
    ModelLoader as MistralModelLoader,
)
from third_party.tt_forge_models.mistral.pytorch.loader import (
    ModelVariant as MistralModelVariant,
)
from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
    ModelLoader as Qwen2_5ModelLoader,
)
from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
    ModelVariant as Qwen2_5ModelVariant,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelVariant as Qwen3ModelVariant,
)

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": Qwen3ModelLoader,
    "bge_m3": BgeModelLoader,
    "bert": BertModelLoader,
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
    "mistral": MistralModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "llama_3_8b",
        "llama_3_8b_instruct",
        "llama_3_1_8b",
        "llama_3_1_8b_instruct",
        "llama_3_1_70b",
        "llama_3_1_70b_instruct",
        "llama_3_1_405b",
        "llama_3_1_405b_instruct",
        "llama_3_2_1b",
        "llama_3_2_1b_instruct",
        "llama_3_2_3b",
        "llama_3_2_3b_instruct",
        "llama_3_3_70b_instruct",
        "huggyllama_7b",
        "TinyLlama_v1.1",
    ],
    "qwen3": ["0_6b", "1_7b", "4b", "8b", "14b", "32b", "30b_a3b"],
    "bge_m3": ["base"],
    "bert": ["bert-base-uncased"],
    "qwen2_5": [
        "0_5b",
        "0_5b_instruct",
        "1_5b",
        "1_5b_instruct",
        "3b",
        "3b_instruct",
        "7b",
        "7b_instruct",
        "7b_instruct_1m",
        "14b",
        "14b_instruct",
        "14b_instruct_1m",
        "32b_instruct",
        "72b_instruct",
        "math_7b",
    ],
    "gemma": [
        "google/gemma-1.1-2b-it",
        "google/gemma-1.1-7b-it",
        "google/gemma-2b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ],
    "mistral": [
        "7b",
        "7b_instruct_v03",
        "ministral_3b_instruct",
        "ministral_8b_instruct",
    ],
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()

    # Filter to only include variants that match names in AVAILABLE_VARIANT_MAP
    if model_name in AVAILABLE_VARIANT_MAP:
        allowed_variant_names = set(AVAILABLE_VARIANT_MAP[model_name])
        return {
            variant_key: variant_config
            for variant_key, variant_config in available_variants.items()
            if str(variant_key) in allowed_variant_names
        }

    return available_variants

# Mark tests to run on both llmbox and single device when shard spec setup is included
def parametrize_is_llmbox():
    return pytest.mark.parametrize(
        "is_llmbox",
        [
            pytest.param(True, marks=pytest.mark.llmbox),
            pytest.param(False, marks=pytest.mark.single_device),
        ],
    )


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
def test_gemma_embeddings(seq_len, is_llmbox):
    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant= GemmaModelVariant.GEMMA_2_27B_IT)
    config = loader.load_config()
    
    class Embeddings(torch.nn.Module):
      def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
      
      def forward(self, input_ids):
        embeddings = self.embed_tokens(input_ids)
        return embeddings

    batch_size = 1
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    embeddings = Embeddings(config)


    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(embeddings, args, kwargs):
            shard_specs = {}
            shard_specs[embeddings.embed_tokens.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        embeddings,
        [
            input_ids,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
def test_gemma_lm_head(seq_len, is_llmbox):
    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant= GemmaModelVariant.GEMMA_2_27B_IT)
    config = loader.load_config()
    
    class LMHead(torch.nn.Module):
      def __init__(self, config):
        super().__init__()
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
      
      def forward(self, input_ids):
        logits = self.lm_head(input_ids)
        return logits

    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    lm_head = LMHead(config)


    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(lm_head, args, kwargs):
            shard_specs = {}
            shard_specs[lm_head.lm_head.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        lm_head,
        [
            hidden_states,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

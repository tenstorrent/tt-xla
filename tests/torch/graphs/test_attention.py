# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from transformers.cache_utils import StaticCache
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gpt_oss.modeling_gpt_oss import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    eager_attention_forward,
)
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaSelfAttention

from tests.utils import parametrize_arch
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
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GPTOSSModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelVariant as GPTOSSModelVariant,
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
    "gpt_oss": GPTOSSModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "3.0_8B",
        "3.1_8B",
        "3.1_70B",
        "3.2_1B",
        "3.2_3B",
        "3.3_70B_Instruct",
        "Huggyllama_7B",
        "Tinyllama_v1.1",
    ],
    "qwen3": ["0_6b", "1_7b", "4b", "8b", "14b", "32b", "30b_a3b"],
    "bge_m3": ["base"],
    "bert": ["bert-base-uncased"],
    "qwen2_5": [
        "0_5b",
        "1_5b",
        "3b",
        "7b",
        "14b",
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
    "gpt_oss": ["gpt_oss_20b", "gpt_oss_120b"],
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


"""Llama attention tests"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_prefill(seq_len, variant, variant_config, arch):
    if "70b" in str(variant) and not arch == "llmbox":
        pytest.skip("70B models don't fit on a single device")

    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = LlamaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1
    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[args[0]] = ("batch", None, None)  # hidden_states
                shard_specs[args[1][0]] = ("batch", None, None)  # cos
                shard_specs[args[1][1]] = ("batch", None, None)  # sin
                shard_specs[args[2]] = ("batch", None, None, None)  # mask
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_decode(variant, variant_config, arch):
    if "70b" in str(variant) and not arch == "llmbox":
        pytest.skip("70B models don't fit on a single device")

    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = LlamaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    seq_len = 1
    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[args[0]] = ("batch", None, None)  # hidden_states
                shard_specs[args[1][0]] = ("batch", None, None)  # cos
                shard_specs[args[1][1]] = ("batch", None, None)  # sin
                shard_specs[args[2]] = ("batch", None, None, None)  # mask
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    run_graph_test(
        attention,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_concat_heads(variant, variant_config, seq_len):
    if "70b" in str(variant):
        pytest.skip("70B models don't fit on a single device")

    def concat_heads(attn_output, input_shape):
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()

    batch_size = 1
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    run_graph_test(concat_heads, [attn_output, input_shape], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_create_heads(variant, variant_config, seq_len):
    if "70b" in str(variant):
        pytest.skip("70B models don't fit on a single device")

    def create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj):
        query_states = q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = LlamaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1
    hidden_size = config.hidden_size
    head_dim = config.head_dim

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    q_proj = attention.q_proj
    k_proj = attention.k_proj
    v_proj = attention.v_proj

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, q_proj, k_proj, v_proj],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention(variant, variant_config, seq_len, arch):
    if "70b" in str(variant) and not arch == "llmbox":
        pytest.skip("70B models don't fit on a single device")

    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):
        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
        )
        return attn_output

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = LlamaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.head_dim

    dropout = 0.0
    scaling = attention.scaling

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)
        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)

        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa, args, kwargs):
            shard_specs = {}
            shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
            shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
            shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
            shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Qwen3 attention tests"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_attention_prefill(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and (str(variant) == "32b" or str(variant) == "30b_a3b"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    if variant == Qwen3ModelVariant.QWEN_3_30B_A3B:
        model = loader.load_model(dtype_override=torch.bfloat16)
        attention = model.model.layers[0].self_attn
    else:
        attention = Qwen3Attention(config, layer_idx=0).to(torch.bfloat16)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        # Qwen3-30B-A3B has 4 key value heads  so it would use 2x4 mesh
        # Thus, we need to see if 2x4 mesh is needed for all Qwen3 models
        num_key_value_heads = config.num_key_value_heads

        if num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Don't shard args - no batch dimension splitting
                # Only shard the model weights
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            # Use 2x4 mesh when num_key_value_heads not divisible by 8 (Qwen3-30B-A3B)
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Shard args on batch dimension
                shard_specs[args[0]] = ("batch", None, None)
                shard_specs[args[1][0]] = ("batch", None, None)
                shard_specs[args[1][1]] = ("batch", None, None)
                shard_specs[args[2]] = ("batch", None, None, None)
                # Shard weights on model dimension
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [Qwen3ModelVariant.QWEN_3_8B])
def test_qwen3_attention_prefill_push(seq_len, variant, arch):
    xr.set_device_type("TT")

    batch_size = 1

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen3Attention(config, layer_idx=0).to(torch.bfloat16)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_attention_decode(variant, variant_config, arch):
    if not arch == "llmbox" and (str(variant) == "32b" or str(variant) == "30b_a3b"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = model.model.layers[0].self_attn

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        # Qwen3-30B-A3B has 4 key value heads  so it would use 2x4 mesh
        # Thus, we need to see if 2x4 mesh is needed for all Qwen3 models
        num_key_value_heads = config.num_key_value_heads

        if num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Don't shard args - no batch dimension splitting
                # Only shard the model weights
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            # Use 2x4 mesh when num_key_value_heads not divisible by 8 (Qwen3-30B-A3B)
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Shard args on batch dimension
                shard_specs[args[0]] = ("batch", None, None)
                shard_specs[args[1][0]] = ("batch", None, None)
                shard_specs[args[1][1]] = ("batch", None, None)
                shard_specs[args[2]] = ("batch", None, None, None)
                # Shard weights on model dimension
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    run_graph_test(
        attention,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_concat_heads(variant, variant_config, seq_len):
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    def concat_heads(attn_output, input_shape):
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()

    batch_size = 1
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    run_graph_test(concat_heads, [attn_output, input_shape], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_create_heads(variant, variant_config, seq_len):
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    def create_heads(
        hidden_states, hidden_shape, q_proj, k_proj, v_proj, q_norm, k_norm
    ):
        query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen3Attention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1
    hidden_size = config.hidden_size
    head_dim = config.head_dim

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    q_proj = attention.q_proj
    k_proj = attention.k_proj
    v_proj = attention.v_proj
    q_norm = attention.q_norm
    k_norm = attention.k_norm

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, q_proj, k_proj, v_proj, q_norm, k_norm],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_attention(variant, variant_config, seq_len, arch):
    if not arch == "llmbox" and (str(variant) == "32b" or str(variant) == "30b_a3b"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):

        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=getattr(attention_module, "sliding_window", None),
        )
        return attn_output

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    if variant == Qwen3ModelVariant.QWEN_3_30B_A3B:
        model = loader.load_model(dtype_override=torch.bfloat16)
        attention = model.model.layers[0].self_attn
    else:
        attention = Qwen3Attention(config, layer_idx=0).to(torch.bfloat16)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        # Qwen3-30B-A3B has 4 key value heads  so it would use 2x4 mesh
        # Thus, we need to see if 2x4 mesh is needed for all Qwen3 models
        num_key_value_heads = config.num_key_value_heads

        if num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        else:
            # Use 2x4 mesh when num_key_value_heads not divisible by 8 (Qwen3-30B-A3B)
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa, args, kwargs):
            shard_specs = {}
            shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
            shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
            shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
            shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
            return shard_specs

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.head_dim

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    dropout = 0.0
    scaling = attention.scaling

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""BGE-M3 attention (XLM-RoBERTa attention) tests"""


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("bge_m3").items(),
    ids=[str(k) for k in get_available_variants("bge_m3").keys()],
)
@pytest.mark.xfail(
    reason="PCC comparison failed with pcc=0.9486. https://github.com/tenstorrent/tt-xla/issues/2214"
)
def test_bge_m3_attention_prefill(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    loader = BgeModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = XLMRobertaSelfAttention(config).to(torch.float32)

    batch_size = 1
    hidden_size = config.hidden_size
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, 1, 1, seq_len), dtype=torch.float32)

    past_key_value = None

    run_graph_test(
        attention,
        [hidden_states, attention_mask, past_key_value],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("bge_m3").items(),
    ids=[str(k) for k in get_available_variants("bge_m3").keys()],
)
def test_bge_m3_concat_heads(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    def concat_heads(context_layer, all_head_size):
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    loader = BgeModelLoader(variant=variant)
    config = loader.load_config()

    batch_size = 1
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    all_head_size = config.hidden_size
    context_layer = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
    )

    run_graph_test(
        concat_heads, [context_layer, all_head_size], framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("bge_m3").items(),
    ids=[str(k) for k in get_available_variants("bge_m3").keys()],
)
def test_bge_m3_create_heads(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    def create_heads(
        hidden_states,
        query_layer,
        key_layer,
        value_layer,
        num_attention_heads,
        attention_head_size,
    ):
        def transpose_for_scores(x):
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        query_states = transpose_for_scores(query_layer(hidden_states))
        key_states = transpose_for_scores(key_layer(hidden_states))
        value_states = transpose_for_scores(value_layer(hidden_states))

        return query_states, key_states, value_states

    loader = BgeModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = XLMRobertaSelfAttention(config).to(torch.float32)

    batch_size = 1
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32)

    query_layer = attention.query
    key_layer = attention.key
    value_layer = attention.value

    run_graph_test(
        create_heads,
        [hidden_states, query_layer, key_layer, value_layer, num_heads, head_dim],
        framework=Framework.TORCH,
    )


"""Bert attention tests"""


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("bert").items(),
    ids=[str(k) for k in get_available_variants("bert").keys()],
)
def test_bert_create_heads(variant, variant_config, seq_len):

    def create_heads(hidden_states, hidden_shape, query_proj, key_proj, value_proj):
        query_states = query_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = key_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = value_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = BertModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = BertSelfAttention(config).to(torch.bfloat16)

    batch_size = 1
    hidden_size = config.hidden_size
    head_dim = attention.attention_head_size

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_proj = attention.query
    key_proj = attention.key
    value_proj = attention.value

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, query_proj, key_proj, value_proj],
        framework=Framework.TORCH,
    )


"""Qwen 2.5 attention tests"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_attention_prefill(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and (
        str(variant) == "72b_instruct" or str(variant) == "32b_instruct"
    ):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).to(torch.bfloat16)

    # Determine batch size and mesh configuration based on attention heads
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Don't shard args - no batch dimension splitting
                # Only shard the model weights
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        elif num_heads % 4 == 0 and num_key_value_heads % 4 == 0:
            # Use 2x4 mesh when divisible by 4
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Shard args on batch dimension
                shard_specs[args[0]] = ("batch", None, None)
                shard_specs[args[1][0]] = ("batch", None, None)
                shard_specs[args[1][1]] = ("batch", None, None)
                shard_specs[args[2]] = ("batch", None, None, None)
                # Shard weights on model dimension
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            pytest.skip("1x8 and 2x4 mesh not supported for this variant")

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [Qwen2_5ModelVariant.QWEN_2_5_7B_INSTRUCT])
def test_qwen2_5_attention_prefill_push(seq_len, variant, arch):
    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).to(torch.bfloat16)

    # Determine batch size and mesh configuration based on attention heads
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Don't shard args - no batch dimension splitting
                # Only shard the model weights
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        elif num_heads % 4 == 0 and num_key_value_heads % 4 == 0:
            # Use 2x4 mesh when divisible by 4
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Shard args on batch dimension
                shard_specs[args[0]] = ("batch", None, None)
                shard_specs[args[1][0]] = ("batch", None, None)
                shard_specs[args[1][1]] = ("batch", None, None)
                shard_specs[args[2]] = ("batch", None, None, None)
                # Shard weights on model dimension
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            pytest.skip("1x8 and 2x4 mesh not supported for this variant")

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_attention_decode(variant, variant_config, arch):
    if not arch == "llmbox" and (
        str(variant) == "72b_instruct" or str(variant) == "32b_instruct"
    ):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).to(torch.bfloat16)

    # Determine batch size and mesh configuration based on attention heads
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Don't shard args - no batch dimension splitting
                # Only shard the model weights
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        elif num_heads % 4 == 0 and num_key_value_heads % 4 == 0:
            # Use 2x4 mesh when divisible by 4
            batch_size = 2
            mesh_shape = (2, num_devices // 2)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                # Shard args on batch dimension
                shard_specs[args[0]] = ("batch", None, None)
                shard_specs[args[1][0]] = ("batch", None, None)
                shard_specs[args[1][1]] = ("batch", None, None)
                shard_specs[args[2]] = ("batch", None, None, None)
                # Shard weights on model dimension
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            pytest.skip("1x8 and 2x4 mesh not supported for this variant")

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    run_graph_test(
        attention,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_attention(variant, variant_config, seq_len, arch):
    if not arch == "llmbox" and (
        str(variant) == "72b_instruct" or str(variant) == "32b_instruct"
    ):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):
        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=getattr(attention_module, "sliding_window", None),
        )
        return attn_output

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).to(torch.bfloat16)

    # Determine batch size and mesh configuration based on attention heads
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            # Use 1x8 mesh for full model parallelism
            batch_size = 1
            mesh_shape = (1, num_devices)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

            def get_shard_spec(sdpa, args, kwargs):
                shard_specs = {}
                # Shard QKV states on model dimension only (not batch)
                shard_specs[args[1]] = (None, "model", None, None)  # query_states
                shard_specs[args[2]] = (None, "model", None, None)  # key_states
                shard_specs[args[3]] = (None, "model", None, None)  # value_states
                shard_specs[args[4]] = (None, None, None, None)  # attention_mask
                return shard_specs

        else:
            if num_heads % 4 == 0 and num_key_value_heads % 4 == 0:
                batch_size = 2
                mesh_shape = (2, num_devices // 2)
                device_ids = np.array(range(num_devices))
                mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

                def get_shard_spec(sdpa, args, kwargs):
                    shard_specs = {}
                    shard_specs[args[1]] = (
                        "batch",
                        "model",
                        None,
                        None,
                    )  # query_states
                    shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
                    shard_specs[args[3]] = (
                        "batch",
                        "model",
                        None,
                        None,
                    )  # value_states
                    shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
                    return shard_specs

            else:
                pytest.skip("1x8 and 2x4 mesh not supported for this variant")

    else:
        batch_size = 1
        mesh = None
        get_shard_spec = None

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_key_value_heads = (
        config.num_key_value_heads
    )  # getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    dropout = 0.0
    scaling = attention.scaling

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Gemma attention tests"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_attention_prefill(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and (str(variant) == "google/gemma-2-27b-it"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = GemmaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1
    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        elif num_heads % 8 == 0 and num_key_value_heads == 1:
            # for small variants like gemma 1.1 2b with 8 attn heads and 1 kv head
            mesh_shape = (1, num_devices)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[args[0]] = ("batch", None, None)  # hidden_states
                shard_specs[args[1][0]] = ("batch", None, None)  # cos
                shard_specs[args[1][1]] = ("batch", None, None)  # sin
                shard_specs[args[2]] = ("batch", None, None, None)  # attention_mask
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.97))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [GemmaModelVariant.GEMMA_2_9B_IT])
def test_gemma_attention_prefill_push(seq_len, variant, arch):
    xr.set_device_type("TT")

    batch_size = 1

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = GemmaAttention(config, layer_idx=0).to(torch.bfloat16)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.97))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_attention_decode(variant, variant_config, arch):
    if not arch == "llmbox" and (str(variant) == "google/gemma-2-27b-it"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = GemmaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1
    seq_len = 1
    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        elif num_heads % 8 == 0 and num_key_value_heads == 1:
            # for small variants like gemma 1.1 2b with 8 attn heads and 1 kv head
            mesh_shape = (1, num_devices)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)

            def get_shard_spec(attention, args, kwargs):
                shard_specs = {}
                shard_specs[args[0]] = ("batch", None, None)  # hidden_states
                shard_specs[args[1][0]] = ("batch", None, None)  # cos
                shard_specs[args[1][1]] = ("batch", None, None)  # sin
                shard_specs[args[2]] = ("batch", None, None, None)  # attention_mask
                shard_specs[attention.q_proj.weight] = ("model", None)
                shard_specs[attention.k_proj.weight] = ("model", None)
                shard_specs[attention.v_proj.weight] = ("model", None)
                shard_specs[attention.o_proj.weight] = (None, "model")
                return shard_specs

        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    run_graph_test(
        attention,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_attention(variant, variant_config, seq_len, arch):
    if not arch == "llmbox" and (str(variant) == "google/gemma-2-27b-it"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):
        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
        )
        return attn_output

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = GemmaAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.head_dim

    dropout = 0.0
    scaling = attention.scaling

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        if num_heads % 8 == 0 and num_key_value_heads % 8 == 0:
            mesh_shape = (1, num_devices)
        else:
            batch_size = 2
            mesh_shape = (2, num_devices // 2)

        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        # replicate key, value states if num_kv_heads=1 like in Gemma 1.1 2b
        if num_key_value_heads == 1:

            def get_shard_spec(sdpa, args, kwargs):
                shard_specs = {}
                shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
                shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
                return shard_specs

        else:

            def get_shard_spec(sdpa, args, kwargs):
                shard_specs = {}
                shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
                shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
                shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
                shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
                return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Mistral attention tests"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_attention_prefill(seq_len, variant, variant_config, arch):
    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = MistralAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [MistralModelVariant.MISTRAL_7B])
def test_mistral_attention_prefill_push(seq_len, variant, arch):
    xr.set_device_type("TT")

    batch_size = 1

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = MistralAttention(config, layer_idx=0).to(torch.bfloat16)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_attention_decode(variant, variant_config, arch):
    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = MistralAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        attention,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_attention(variant, variant_config, seq_len, arch):
    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):

        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=getattr(attention_module, "sliding_window", None),
        )
        return attn_output

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    config._attn_implementation = "sdpa"
    attention = MistralAttention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 1

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    dropout = 0.0
    scaling = attention.scaling

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa, args, kwargs):
            shard_specs = {}
            shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
            shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
            shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
            shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


class MinimalAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose
        query_states = query_states.view(
            batch_size, seq_len, -1, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            batch_size, seq_len, -1, self.head_dim
        ).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output


@pytest.mark.push
@pytest.mark.llmbox
def test_compiled_batched_attention():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}
        shard_specs[model.q_proj.weight] = ("model", None)
        shard_specs[model.k_proj.weight] = ("model", None)
        shard_specs[model.v_proj.weight] = ("model", None)
        shard_specs[model.o_proj.weight] = (None, "model")
        return shard_specs

    # Model config
    hidden_size = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    model = MinimalAttention(hidden_size, num_heads).to(torch.bfloat16)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)
    run_graph_test(
        model,
        [
            hidden_states,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.push
@pytest.mark.llmbox
def test_eager_batched_attention():
    # import tt_torch
    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    # Model config
    hidden_size = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    device: torch.device = torch_xla.device()

    model = MinimalAttention(hidden_size, num_heads).to(torch.bfloat16).to(device)

    xs.mark_sharding(model.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(model.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(model.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(model.o_proj.weight, mesh, (None, "model"))
    hidden_states = (
        torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16).to(device)
    )

    output = model(hidden_states).to("cpu")


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
@parametrize_arch(["single_device", "llmbox"])
def test_gpt_oss_attention(variant, variant_config, arch):
    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):
        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
        )
        return attn_output

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]

    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.head_dim

    attention = model.model.layers[0].self_attn

    dropout = 0.0
    scaling = attention.scaling

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        mesh_shape = (1, num_devices)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa, args, kwargs):
            # Attention is the first argument to sdpa.
            attention = args[0]
            shard_specs = {}
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.97))

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Deepseek attention tests"""


import scipy.linalg  # for hadamard matrix

from tests.torch.models.deepseek_v3_2_exp.modified_model import MLA as DeepseekMLA
from tests.torch.models.deepseek_v3_2_exp.modified_model import (
    ModelArgs as DeepseekModelArgs,
)


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
def test_deepseek_attention_module(seq_len, arch):
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = DeepseekModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    # Generate Hadamard matrix once and register as a buffer so it moves with the model
    hidden_size = args.index_head_dim

    hadamard_matrix = torch.tensor(
        scipy.linalg.hadamard(hidden_size), dtype=torch.bfloat16
    ) * (hidden_size**-0.5)
    attention = DeepseekMLA(args, hadamard_matrix).to(torch.bfloat16)

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)
    start_pos = 0
    # Shape: (seq_len, qk_rope_head_dim // 2)
    # Type: complex64 (standard for RoPE)
    freqs_cis = torch.randn(
        seq_len, args.qk_rope_head_dim // 2, 2, dtype=torch.float32
    )  # double check dtype
    mask = torch.rand(batch_size, seq_len, seq_len, dtype=torch.bfloat16)

    # Setup for tensor parallel
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        # TODO: double check shard specs
        def get_shard_spec(attention, args, kwargs):
            """Returns shard specifications for the layer's parameters."""
            shard_specs = {}

            # Down projections (Compression) - Replicated
            # These are standard Linear layers, so we map them to "batch" (replicated)
            shard_specs[attention.wq_a.weight] = ("batch", "batch")

            shard_specs[attention.wkv_a.weight] = ("batch", "batch")

            # Up projections (Decompression to Heads) - Column Parallel
            # Shard the output dimension (dim 0)
            shard_specs[attention.wq_b.weight] = ("model", "batch")

            shard_specs[attention.wkv_b.weight] = ("model", "batch")

            # Output projection - Row Parallel
            # Shard the input dimension (dim 1)
            shard_specs[attention.wo.weight] = ("batch", "model")

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        attention,
        [hidden_states, start_pos, freqs_cis, mask],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Kimi K2 attention tests"""

import os
import sys

model_dir = os.path.join(os.path.dirname(__file__), "../models/kimi_k2")
sys.path.append(os.path.abspath(model_dir))

from tests.torch.models.kimi_k2.configuration_deepseek import DeepseekV3Config
from tests.torch.models.kimi_k2.modeling_deepseek import (
    DeepseekV3Attention as KimiK2Attention,
)


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
def test_kimi_k2_attention_module(seq_len, arch):
    xr.set_device_type("TT")

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "../models/kimi_k2/config.json")

    config = DeepseekV3Config.from_json_file(config_path)

    # Override for single layer testing
    config.num_hidden_layers = 1
    config.use_cache = False

    attention = KimiK2Attention(config, layer_idx=0).to(torch.bfloat16)

    batch_size = 2
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    position_ids = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.long)

    past_key_value = None

    # Setup for tensor parallel
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            """Returns shard specifications for the layer's parameters."""
            shard_specs = {}

            # TODO: Add layer norm shard specs too
            shard_specs[attention.q_a_proj.weight] = ("model", "batch")
            shard_specs[attention.q_b_proj.weight] = ("model", "batch")
            shard_specs[attention.kv_a_proj_with_mqa.weight] = ("model", "batch")
            shard_specs[attention.kv_b_proj.weight] = ("model", "batch")
            shard_specs[attention.o_proj.weight] = ("batch", "model")

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        attention,
        [hidden_states, attention_mask, position_ids, past_key_value, False, False],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

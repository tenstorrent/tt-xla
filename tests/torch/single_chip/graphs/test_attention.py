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
from infra.comparators.torch_comparator import TorchComparator
from torch_xla.distributed.spmd import Mesh
from transformers import CacheConfig
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
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
from third_party.tt_forge_models.qwen_2_5.casual_lm.pytorch.loader import (
    ModelLoader as Qwen2_5ModelLoader,
)
from third_party.tt_forge_models.qwen_2_5.casual_lm.pytorch.loader import (
    ModelVariant as Qwen2_5ModelVariant,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as QwenModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelVariant as Qwen3ModelVariant,
)

# To see all available models and variants, run:
# pytest -s tests/torch/single_chip/graphs/test_attention.py::test_display_available_variants

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": QwenModelLoader,
    "bge_m3": BgeModelLoader,
    "bert": BertModelLoader,
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()
    return available_variants


@pytest.mark.parametrize("model_name", list(MODEL_LOADER_MAP.keys()))
def test_display_available_variants(model_name):
    print(
        f"\nAvailable variants for {model_name}: ",
        [str(k) for k in get_available_variants(model_name)],
    )


"""Llama attention tests"""


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_prefill(seq_len, variant, variant_config):
    # Xfail 70B models that don't fit on device
    if "70b" in str(variant):
        pytest.xfail("70B models don't fit on device")

    # Will download huge amount of data and run out of disk space.
    if "405b" in str(variant):
        pytest.skip("405B variants too large for device and disk space")

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    hidden_states = torch.randn(
        (1, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(1, seq_len, model.config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_decode(variant, variant_config):

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    seq_len = 1
    hidden_states = torch.randn(
        (1, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(1, seq_len, model.config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    batch_size = 1
    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_concat_heads(variant, variant_config, seq_len):

    if str(variant) == "llama_3_1_405b" or str(variant) == "llama_3_1_405b_instruct":
        pytest.xfail("Variant doesn't fit on device")

    def concat_heads(attn_output, input_shape):
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    batch_size = 1
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    run_graph_test(concat_heads, [attn_output, input_shape], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_create_heads(variant, variant_config, seq_len):

    def create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj):
        query_states = q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

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
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_sdpa(variant, variant_config, seq_len):

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
        return attn_output, attn_weights

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.head_dim

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

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
    )


"""Qwen3 attention tests"""


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_attention_prefill(seq_len, variant, variant_config, request):
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

    xr.set_device_type("TT")

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(
        batch_size, seq_len, model.config.head_dim, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
@pytest.mark.parametrize(
    "is_llmbox",
    [
        pytest.param(True, marks=pytest.mark.llmbox),
        pytest.param(False, marks=pytest.mark.single_device),
    ],
)
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [Qwen3ModelVariant.QWEN_3_8B])
def test_qwen3_attention_prefill_push(seq_len, variant, is_llmbox):
    xr.set_device_type("TT")

    if is_llmbox:
        batch_size = 2
    else:
        batch_size = 1

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(
        batch_size, seq_len, model.config.head_dim, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_attention_decode(variant, variant_config):
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

    xr.set_device_type("TT")

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    seq_len = 1
    hidden_states = torch.randn(
        (1, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(1, seq_len, model.config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    batch_size = 1
    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.tensor([0])
    past_key_states = static_cache

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_concat_heads(variant, variant_config, seq_len):
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

    xr.set_device_type("TT")

    def concat_heads(attn_output, input_shape):
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    batch_size = 1
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    run_graph_test(concat_heads, [attn_output, input_shape], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_create_heads(variant, variant_config, seq_len):
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

    xr.set_device_type("TT")

    def create_heads(
        hidden_states, hidden_shape, q_proj, k_proj, v_proj, q_norm, k_norm
    ):
        query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

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
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_sdpa(variant, variant_config, seq_len):
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

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
            sliding_window=getattr(attention_module, "sliding_window", None),
        )
        return attn_output, attn_weights

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.head_dim

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

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
    )


"""Qwen 2.5 attention tests"""


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_attention_prefill(seq_len, variant, variant_config, request):
    if str(variant) == "72b_instruct":
        pytest.xfail("Not enough memory to run this test")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@pytest.mark.parametrize(
    "is_llmbox",
    [
        pytest.param(True, marks=pytest.mark.llmbox),
        pytest.param(False, marks=pytest.mark.single_device),
    ],
)
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [Qwen2_5ModelVariant.QWEN_2_5_7B_INSTRUCT])
def test_qwen2_5_attention_prefill_push(seq_len, variant, is_llmbox):
    xr.set_device_type("TT")

    if is_llmbox:
        batch_size = 2
    else:
        batch_size = 1

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_attention_decode(variant, variant_config, request):
    if str(variant) == "72b_instruct":
        pytest.xfail("Not enough memory to run this test")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.tensor([0])
    past_key_states = static_cache

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
            shard_specs[attention.q_proj.weight] = ("model", None)
            shard_specs[attention.k_proj.weight] = ("model", None)
            shard_specs[attention.v_proj.weight] = ("model", None)
            shard_specs[attention.o_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_concat_heads(variant, variant_config, seq_len, request):
    if str(variant) == "72b_instruct":
        pytest.xfail("Not enough memory to run this test")

    xr.set_device_type("TT")

    def concat_heads(attn_output, input_shape):
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(concat_heads_fn, args, kwargs):
            shard_specs = {}
            # attn_output is sharded on batch and num_heads (model) dimensions
            shard_specs[args[0]] = ("batch", "model", None, None)
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        concat_heads,
        [attn_output, input_shape],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_create_heads(variant, variant_config, seq_len, request):
    if str(variant) == "72b_instruct":
        pytest.xfail("Not enough memory to run this test")

    xr.set_device_type("TT")

    def create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj):
        query_states = q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_size = model.config.hidden_size
    head_dim = hidden_size // model.config.num_attention_heads

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    # q_proj = attention.q_proj
    # k_proj = attention.k_proj
    # v_proj = attention.v_proj

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(create_heads, args, kwargs):
            shard_specs = {}
            # Input: hidden_states is sharded on batch and hidden_size (model) dimensions
            shard_specs[args[0]] = ("batch", None, "model")
            # shard_specs[args[1]] = ("batch", None, "model") # Does not need to be sharded
            # Projection weights are sharded on output dimension (which becomes heads)
            shard_specs[args[2].weight] = ("model", None)
            shard_specs[args[3].weight] = ("model", None)
            shard_specs[args[4].weight] = ("model", None)
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        create_heads,
        [
            hidden_states,
            hidden_shape,
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_sdpa(variant, variant_config, seq_len, request):
    if str(variant) == "72b_instruct":
        pytest.xfail("Not enough memory to run this test")

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
            sliding_window=getattr(attention_module, "sliding_window", None),
        )
        return attn_output, attn_weights

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_key_value_heads = (
        model.config.num_key_value_heads
    )  # getattr(model.config, "num_key_value_heads", num_heads)
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

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa_fn, args, kwargs):
            shard_specs = {}
            # args[0] is attention module - no sharding needed
            # Query, key, value states sharded on batch and heads dimensions
            shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
            shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
            shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
            shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
            # args[5] is dropout (scalar), args[6] is scaling (scalar) - no sharding
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

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
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
    # xm.mark_step()
    torch_xla.sync()
    xm.wait_device_ops()


"""BGE-M3 attention (XLM-RoBERTa attention) tests"""


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("bge_m3").items(),
    ids=[str(k) for k in get_available_variants("bge_m3").keys()],
)
@pytest.mark.skip(
    reason=failed_runtime(
        "Test hangs - https://github.com/tenstorrent/tt-xla/issues/1830"
    )
)
def test_bge_m3_attention_prefill(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    loader = BgeModelLoader(variant=variant)
    model = loader.load_model()
    attention = model.model.encoder.layer[0].attention

    batch_size = 1
    hidden_size = model.config.hidden_size
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, 1, 1, seq_len), dtype=torch.float32)

    past_key_value = None

    run_graph_test(
        attention,
        [hidden_states, attention_mask, past_key_value],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
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
    model = loader.load_model()

    batch_size = 1
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    all_head_size = model.config.hidden_size
    context_layer = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
    )

    run_graph_test(
        concat_heads, [context_layer, all_head_size], framework=Framework.TORCH
    )


@pytest.mark.nightly
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
    model = loader.load_model()
    attention = model.model.encoder.layer[0].attention.self

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32)

    query_layer = attention.query
    key_layer = attention.key
    value_layer = attention.value

    run_graph_test(
        create_heads,
        [hidden_states, query_layer, key_layer, value_layer, num_heads, head_dim],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
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
    model = loader.load_model(dtype_override=torch.bfloat16)

    attention = model.bert.encoder.layer[0].attention.self

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
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


"""Gemma attention tests"""


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_attention_prefill(seq_len, variant, variant_config, request):
    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(
        batch_size, seq_len, model.config.head_dim, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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


# Add single push test to ensure multi-chip graph tester has coverage.
@pytest.mark.push
@pytest.mark.parametrize(
    "is_llmbox",
    [
        pytest.param(True, marks=pytest.mark.llmbox),
        pytest.param(False, marks=pytest.mark.single_device),
    ],
)
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", [GemmaModelVariant.GEMMA_2_9B_IT])
def test_gemma_attention_prefill_push(seq_len, variant, is_llmbox):
    xr.set_device_type("TT")

    if is_llmbox:
        batch_size = 2
    else:
        batch_size = 1

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(
        batch_size, seq_len, model.config.head_dim, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_attention_decode(variant, variant_config, request):
    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(
        batch_size, seq_len, model.config.head_dim, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1][0]] = ("batch", None, None)
            shard_specs[args[1][1]] = ("batch", None, None)
            shard_specs[args[2]] = ("batch", None, None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_concat_heads(variant, variant_config, seq_len, request):
    xr.set_device_type("TT")

    def concat_heads(attn_output, input_shape):
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(concat_heads_fn, args, kwargs):
            shard_specs = {}
            # attn_output is sharded on batch and num_heads (model) dimensions
            shard_specs[args[0]] = ("batch", "model", None, None)
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        concat_heads,
        [attn_output, input_shape],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_create_heads(variant, variant_config, seq_len, request):
    xr.set_device_type("TT")

    def create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj):
        query_states = q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(create_heads, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[2].weight] = ("model", None)
            shard_specs[args[3].weight] = ("model", None)
            shard_specs[args[4].weight] = ("model", None)
            if args[2].bias is not None:
                shard_specs[args[2].bias] = ("model",)
            if args[3].bias is not None:
                shard_specs[args[3].bias] = ("model",)
            if args[4].bias is not None:
                shard_specs[args[4].bias] = ("model",)
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        create_heads,
        [
            hidden_states,
            hidden_shape,
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_sdpa(variant, variant_config, seq_len, request):
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
        return attn_output, attn_weights

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.head_dim

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

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(sdpa_fn, args, kwargs):
            shard_specs = {}
            # args[0] is attention module - no sharding needed
            # Query, key, value states sharded on batch and heads dimensions
            shard_specs[args[1]] = ("batch", "model", None, None)  # query_states
            shard_specs[args[2]] = ("batch", "model", None, None)  # key_states
            shard_specs[args[3]] = ("batch", "model", None, None)  # value_states
            shard_specs[args[4]] = ("batch", None, None, None)  # attention_mask
            # args[5] is dropout (scalar), args[6] is scaling (scalar) - no sharding
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

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
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

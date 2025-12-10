# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from tests.utils import parametrize_arch
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelLoader as GemmaModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelVariant as GemmaModelVariant,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelVariant as LlamaModelVariant,
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
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
    "mistral": MistralModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "llama_3_8b",
        "llama_3_1_8b",
        "llama_3_1_70b",
        "llama_3_2_1b",
        "llama_3_2_3b",
        "llama_3_3_70b_instruct",
        "huggyllama_7b",
        "TinyLlama_v1.1",
    ],
    "qwen3": ["0_6b", "1_7b", "4b", "8b", "14b", "32b"],
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


"""Llama decode layer test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_decode_layer(seq_len, variant, variant_config, arch):
    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_layer = LlamaDecoderLayer(config, layer_idx=0).to(
                torch.bfloat16
            )

        def forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_state
        ):
            hidden_states = self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_state=past_key_state,
            )
            return hidden_states[0]

    wrapper = Wrapper().to(torch.bfloat16)
    batch_size = 1

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        batch_size = 2
        mesh_shape = (batch_size, num_devices // batch_size)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)  # hidden_states
            shard_specs[args[1][0]] = ("batch", None, None)  # cos
            shard_specs[args[1][1]] = ("batch", None, None)  # sin
            shard_specs[args[2]] = ("batch", None, None, None)  # mask
            shard_specs[wrapper.decoder_layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[wrapper.decoder_layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.rand(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        wrapper,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Qwen3 decode layer test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_decode_layer(seq_len, variant, variant_config, arch):
    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_layer = Qwen3DecoderLayer(config, layer_idx=0).to(
                torch.bfloat16
            )

        def forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_state
        ):
            hidden_states = self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_state=past_key_state,
            )
            return hidden_states[0]

    wrapper = Wrapper().to(torch.bfloat16)
    batch_size = 1

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        batch_size = 2
        mesh_shape = (batch_size, num_devices // batch_size)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)  # hidden_states
            shard_specs[args[1][0]] = ("batch", None, None)  # cos
            shard_specs[args[1][1]] = ("batch", None, None)  # sin
            shard_specs[args[2]] = ("batch", None, None, None)  # mask
            shard_specs[wrapper.decoder_layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[wrapper.decoder_layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.rand(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        wrapper,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Gemma decode layer test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_decode_layer(seq_len, variant, variant_config, arch):
    if str(variant) == "google/gemma-1.1-2b-it" or str(variant) == "google/gemma-2b":
        pytest.skip("Only running variants that support 2x4 sharding")

    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()

    # For Gemma2 use the Gemma2DecoderLayer
    if "gemma-2-" in str(variant):

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_layer = Gemma2DecoderLayer(config, layer_idx=0).to(
                    torch.bfloat16
                )

            def forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_state,
                position_ids,
                cache_position,
            ):
                hidden_states = self.decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_state,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                )
                return hidden_states[0]

    else:

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_layer = GemmaDecoderLayer(config, layer_idx=0).to(
                    torch.bfloat16
                )

            def forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_state,
                position_ids,
                cache_position,
            ):
                hidden_states = self.decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_state=past_key_state,
                )
                return hidden_states[0]

    wrapper = Wrapper().to(torch.bfloat16)
    batch_size = 1

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        batch_size = 2
        mesh_shape = (batch_size, num_devices // batch_size)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)  # hidden_states
            shard_specs[args[1][0]] = ("batch", None, None)  # cos
            shard_specs[args[1][1]] = ("batch", None, None)  # sin
            shard_specs[args[2]] = ("batch", None, None, None)  # mask
            shard_specs[wrapper.decoder_layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[wrapper.decoder_layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.rand(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    cos_sin = torch.rand(batch_size, seq_len, config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    past_key_states = None

    # For Gemma2, we need to provide position_ids and cache_position
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        wrapper,
        [
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_states,
            position_ids,
            cache_position,
        ],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Mistral decode layer test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_decode_layer(seq_len, variant, variant_config, arch):
    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_layer = MistralDecoderLayer(config, layer_idx=0).to(
                torch.bfloat16
            )

        def forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_state
        ):
            hidden_states = self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_state=past_key_state,
            )
            return hidden_states[0]

    wrapper = Wrapper().to(torch.bfloat16)
    batch_size = 1

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        batch_size = 2
        mesh_shape = (batch_size, num_devices // batch_size)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)  # hidden_states
            shard_specs[args[1][0]] = ("batch", None, None)  # cos
            shard_specs[args[1][1]] = ("batch", None, None)  # sin
            shard_specs[args[2]] = ("batch", None, None, None)  # mask
            shard_specs[wrapper.decoder_layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[wrapper.decoder_layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.rand(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        wrapper,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Qwen2.5 decode layer test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_decode_layer(seq_len, variant, variant_config, arch):
    if str(variant) == "0_5b" or str(variant) == "1_5b" or str(variant) == "3b":
        pytest.skip("Only running variants that support 2x4 sharding")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_layer = Qwen2DecoderLayer(config, layer_idx=0).to(
                torch.bfloat16
            )

        def forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_state
        ):
            hidden_states = self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_state=past_key_state,
            )
            return hidden_states[0]

    wrapper = Wrapper().to(torch.bfloat16)
    batch_size = 1

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        batch_size = 2
        mesh_shape = (batch_size, num_devices // batch_size)
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)  # hidden_states
            shard_specs[args[1][0]] = ("batch", None, None)  # cos
            shard_specs[args[1][1]] = ("batch", None, None)  # sin
            shard_specs[args[2]] = ("batch", None, None, None)  # mask
            shard_specs[wrapper.decoder_layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[wrapper.decoder_layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[wrapper.decoder_layer.mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    hidden_states = torch.rand(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cos_sin = torch.rand(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        wrapper,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

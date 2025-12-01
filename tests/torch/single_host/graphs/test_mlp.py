# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from transformers.models.falcon.modeling_falcon import FalconMLP
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from third_party.tt_forge_models.falcon.pytorch.loader import (
    ModelLoader as FalconModelLoader,
)
from third_party.tt_forge_models.falcon.pytorch.loader import (
    ModelVariant as FalconModelVariant,
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
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
    "mistral": MistralModelLoader,
    "falcon": FalconModelLoader,
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
    "falcon": [
        "tiiuae/Falcon3-1B-Base",
        "tiiuae/Falcon3-3B-Base",
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
        "tiiuae/Falcon3-Mamba-7B-Base",
        "tiiuae/falcon-7b-instruct",
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


"""Qwen3 MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_mlp(seq_len, variant, variant_config, is_llmbox):
    if not is_llmbox and str(variant) == "32b":
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    mlp = Qwen3MLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            # check if model is a MoE model (Qwen3-30B-A3B)
            if hasattr(mlp, "num_experts"):
                for expert in mlp.experts:
                    shard_specs[expert.gate_proj.weight] = ("model", None)
                    shard_specs[expert.up_proj.weight] = ("model", None)
                    shard_specs[expert.down_proj.weight] = (None, "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", None)
                shard_specs[mlp.up_proj.weight] = ("model", None)
                shard_specs[mlp.down_proj.weight] = (None, "model")

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Llama MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_mlp(seq_len, variant, variant_config, is_llmbox):
    if "70b" in str(variant) and not is_llmbox:
        pytest.skip("70B models don't fit on a single device")

    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    mlp = LlamaMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Gemma MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_mlp(seq_len, variant, variant_config, is_llmbox):
    if not is_llmbox and (str(variant) == "google/gemma-2-27b-it"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    mlp = GemmaMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Mistral MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_mlp(seq_len, variant, variant_config, is_llmbox):

    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    mlp = MistralMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Qwen2_5 MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_mlp(seq_len, variant, variant_config, is_llmbox):
    if not is_llmbox and (
        str(variant) == "72b_instruct" or str(variant) == "32b_instruct"
    ):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    mlp = Qwen2MLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Falcon MLP test"""


@pytest.mark.nightly
@parametrize_is_llmbox()  # True for llmbox, False for single device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("falcon").items(),
    ids=[str(k) for k in get_available_variants("falcon").keys()],
)
def test_falcon_mlp(seq_len, variant, variant_config, is_llmbox):
    if variant != FalconModelVariant.FALCON_7B_INSTRUCT:
        if variant == FalconModelVariant.FALCON_MAMBA_7B:
            pytest.skip("FalconMamba has no MLP as it is a State Space Model.")
        else:
            pytest.skip(
                "Falcon3-Base models use Llama3 architecure, use Llama MLP test instead."
            )

    xr.set_device_type("TT")

    loader = FalconModelLoader(variant=variant)
    config = loader.load_config()
    mlp = FalconMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.dense_h_to_4h.weight] = ("model", None)  # up_proj
            shard_specs[mlp.dense_4h_to_h.weight] = (None, "model")  # down_proj
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

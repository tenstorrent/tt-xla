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

from tests.utils import is_llmbox
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

# To see all available models and variants, run:
# pytest -s tests/torch/single_chip/graphs/test_mlp.py::test_display_available_variants

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": Qwen3ModelLoader,
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
    "mistral": MistralModelLoader,
    "falcon": FalconModelLoader,
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


"""Qwen3 MLP test"""


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_mlp(seq_len, variant, variant_config, request):
    if str(variant) == "32b" or str(variant) == "30b_a3b":
        pytest.xfail("Variant doesn't fit on device")

    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.model.layers[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_mlp(seq_len, variant, variant_config, request):
    # Xfail 70B models that don't fit on device
    if "70b" in str(variant):
        pytest.xfail("70B models don't fit on device")

    # Will download huge amount of data and run out of disk space.
    if "405b" in str(variant):
        pytest.skip("405B variants too large for device and disk space")

    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.model.layers[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_mlp(seq_len, variant, variant_config, request):
    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.model.layers[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_mlp(seq_len, variant, variant_config, request):

    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.model.layers[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_mlp(seq_len, variant, variant_config, request):
    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.model.layers[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("falcon").items(),
    ids=[str(k) for k in get_available_variants("falcon").keys()],
)
def test_falcon_mlp(seq_len, variant, variant_config, request):
    if variant != FalconModelVariant.FALCON_7B_INSTRUCT:
        if variant == FalconModelVariant.FALCON_MAMBA_7B:
            pytest.xfail("FalconMamba has no MLP as it is a State Space Model.")
        else:
            pytest.xfail(
                "Falcon3-Base models use Llama3 architecure, use Llama MLP test instead."
            )

    xr.set_device_type("TT")

    loader = FalconModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    mlp = model.transformer.h[0].mlp

    if is_llmbox(request):
        batch_size = 2
    else:
        batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )

    if is_llmbox(request):
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, num_devices // 2)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[args[0]] = ("batch", None, None)
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

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
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelVariant as LlamaModelVariant,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelVariant as Qwen3ModelVariant,
)

# To see all available models and variants, run:
# pytest -s tests/torch/single_chip/graphs/test_attention.py::test_display_available_variants

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": Qwen3ModelLoader,
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
    if str(variant) == "qwq_32b":
        pytest.xfail("QWQ_32B varaiant is actually Qwen2, which has a different config")
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

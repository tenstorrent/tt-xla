# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "llama_3_8b",
        "llama_3_1_8b",
        "llama_3_2_1b",
        "llama_3_2_3b",
        "huggyllama_7b",
        "TinyLlama_v1.1",
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


"""Llama rms norm test"""


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_rms_norm(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    rms_norm = LlamaRMSNorm(config.hidden_size).to(torch.bfloat16)

    batch_size = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        rms_norm,
        [hidden_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )

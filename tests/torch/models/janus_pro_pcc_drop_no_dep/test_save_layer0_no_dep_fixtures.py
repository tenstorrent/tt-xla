# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU-only: capture real ImageTokenStep decode inputs for no-dep layer-0 sanity.

Writes under ``janus_logs/layer0_tensors/<variant>/`` (override with ``JANUS_LAYER0_TENSOR_DIR``):

- ``inputs_embeds_decode.pt``, ``past_key_values_decode.pt`` (pre-decode KV snapshot)
- ``rotary_emb.pt``, ``input_layernorm.pt``, ``self_attn.pt``, ``llama_config.pt``
- ``hidden_before/after_input_layernorm_layer0.pt``, ``manifest.json``

**One-time** capture (uses Janus loader). After this, sanity/codegen need only
``torch``, ``transformers``, and the saved files under ``janus_logs/layer0_tensors/<variant>/``.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from tests.runner.requirements import RequirementsManager
from tests.torch.models.janus_pro_pcc_drop.layer0_tensor_capture import (
    resolve_variant_layer0_tensor_dir,
    save_layer0_input_layernorm_tensors_from_decode_step,
)

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _save_no_dep_fixtures(variant_name: str) -> None:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()

        loader = ModelLoader(ModelVariant(variant_name))
        decode_inputs = loader.load_inputs(dtype_override=torch.bfloat16, prefill=False)
        step = loader.load_model(dtype_override=torch.bfloat16)

        out_dir = resolve_variant_layer0_tensor_dir(
            variant_name,
            os.environ.get("JANUS_LAYER0_TENSOR_DIR"),
        )

        save_layer0_input_layernorm_tensors_from_decode_step(
            step,
            decode_inputs,
            out_dir,
            variant_label=variant_name,
        )


@pytest.mark.model_test
def test_save_layer0_no_dep_fixtures_pro_1b():
    """CPU-only capture for Pro-1B (same loader path as ``test_image_token_decode_pro_1b``)."""
    _save_no_dep_fixtures("Pro_1B")


@pytest.mark.model_test
def test_save_layer0_no_dep_fixtures_pro_7b():
    """CPU-only capture for Pro-7B."""
    _save_no_dep_fixtures("Pro_7B")

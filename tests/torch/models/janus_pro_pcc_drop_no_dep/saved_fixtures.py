# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load CPU-captured decode fixtures for no-dep layer-0 LN+attn sanity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tests.torch.models.janus_pro_pcc_drop.kv_portable_export import (
    CONFIG_JSON_FILENAME,
    PORTABLE_KV_FILENAME,
    load_past_key_values_portable,
)
from tests.torch.models.janus_pro_pcc_drop.layer0_tensor_capture import (
    SAVED_TENSOR_FILENAMES,
    SAVED_WEIGHT_FILENAMES,
    layer0_fixtures_complete,
    resolve_variant_layer0_tensor_dir,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.constants import DTYPE


def fixture_dir_for_variant(variant: str) -> Path:
    return resolve_variant_layer0_tensor_dir(variant)


def saved_fixtures_available(variant: str) -> bool:
    return layer0_fixtures_complete(fixture_dir_for_variant(variant))


def load_saved_decode_inputs(variant: str) -> tuple[torch.Tensor, Any]:
    """
  Load ``inputs_embeds`` and ``past_key_values`` from
  ``save_layer0_no_dep_fixtures`` (same tensors as ImageTokenStep decode).
  """
    root = fixture_dir_for_variant(variant)
    if not saved_fixtures_available(variant):
        raise FileNotFoundError(
            f"Missing no-dep fixtures under {root}. Run:\n"
            "  pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/"
            "test_save_layer0_no_dep_fixtures.py"
        )
    inputs_embeds = torch.load(
        root / SAVED_TENSOR_FILENAMES["inputs_embeds_decode"],
        weights_only=True,
    )
    portable_kv = root / PORTABLE_KV_FILENAME
    if not portable_kv.is_file():
        raise FileNotFoundError(
            f"Missing {PORTABLE_KV_FILENAME}. Re-run test_save_layer0_no_dep_fixtures "
            "or export_portable_fixtures.py on tt-metal."
        )
    past_key_values = load_past_key_values_portable(portable_kv)
    return inputs_embeds, past_key_values


def apply_saved_layer0_weights(
    bundle: Any,
    variant: str,
    *,
    dtype: torch.dtype = DTYPE,
) -> None:
    """Load ``rotary_emb`` / ``input_layernorm`` / ``self_attn`` + config from fixture dir."""
    root = fixture_dir_for_variant(variant)
    bundle.rotary_emb.load_state_dict(
        torch.load(root / SAVED_WEIGHT_FILENAMES["rotary_emb"], weights_only=True),
        strict=True,
    )
    bundle.input_layernorm.load_state_dict(
        torch.load(root / SAVED_WEIGHT_FILENAMES["input_layernorm"], weights_only=True),
        strict=True,
    )
    bundle.self_attn.load_state_dict(
        torch.load(root / SAVED_WEIGHT_FILENAMES["self_attn"], weights_only=True),
        strict=True,
    )
    json_cfg = root / CONFIG_JSON_FILENAME
    if json_cfg.is_file():
        import json

        from transformers import LlamaConfig

        bundle.llama_config = LlamaConfig.from_dict(json.loads(json_cfg.read_text()))
    else:
        bundle.llama_config = torch.load(
            root / SAVED_WEIGHT_FILENAMES["llama_config"], weights_only=False
        )
    bundle.llama_config.attn_implementation = "eager"
    bundle.llama_config._attn_implementation = "eager"
    for module in (bundle.rotary_emb, bundle.input_layernorm, bundle.self_attn):
        module.to(dtype=dtype)
        module.eval()

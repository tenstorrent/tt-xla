# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Save layer-0 layernorm boundary tensors from the real ImageTokenStep decode forward (CPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DEFAULT_LAYER0_TENSOR_DIR = (
    Path(__file__).resolve().parents[4] / "janus_logs" / "layer0_tensors"
)

SAVED_TENSOR_FILENAMES = {
    "before_input_layernorm": "hidden_before_input_layernorm_layer0.pt",
    "after_input_layernorm": "hidden_after_input_layernorm_layer0.pt",
    "inputs_embeds_decode": "inputs_embeds_decode.pt",
}


def resolve_layer0_tensor_dir(tensor_dir: str | Path | None = None) -> Path:
    import os

    if tensor_dir is not None:
        return Path(tensor_dir)
    return Path(os.environ.get("JANUS_LAYER0_TENSOR_DIR", str(DEFAULT_LAYER0_TENSOR_DIR)))


def load_saved_layer0_tensors(tensor_dir: str | Path | None = None) -> dict[str, torch.Tensor]:
    """Load tensors written by ``save_layer0_input_layernorm_tensors_from_decode_step``."""
    root = resolve_layer0_tensor_dir(tensor_dir)
    return {
        "hidden_before_input_layernorm": torch.load(
            root / SAVED_TENSOR_FILENAMES["before_input_layernorm"], weights_only=True
        ),
        "hidden_after_input_layernorm": torch.load(
            root / SAVED_TENSOR_FILENAMES["after_input_layernorm"], weights_only=True
        ),
        "inputs_embeds_decode": torch.load(
            root / SAVED_TENSOR_FILENAMES["inputs_embeds_decode"], weights_only=True
        ),
    }


@torch.inference_mode()
def save_layer0_input_layernorm_tensors_from_decode_step(
    step: nn.Module,
    decode_inputs: dict[str, Any],
    output_dir: str | Path = DEFAULT_LAYER0_TENSOR_DIR,
    *,
    layer_idx: int = 0,
    variant_label: str = "Pro_1B",
) -> Path:
    """
    Run ``JanusGitImageTokenStep`` on CPU and save hidden states at layer-0 ``input_layernorm``.

    Files written under ``output_dir``:

    - ``hidden_before_input_layernorm_layer0.pt`` — input to ``input_layernorm``
    - ``hidden_after_input_layernorm_layer0.pt`` — output of ``input_layernorm``
    - ``inputs_embeds_decode.pt`` — decode-step embeds passed to the step
    - ``manifest.json`` — shapes / dtypes / repo variant
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llama_model = step.mmgpt.language_model.model
    input_layernorm = llama_model.layers[layer_idx].input_layernorm

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["before"] = args[0].detach().cpu().clone()

    def _post_hook(
        _module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured["after"] = output.detach().cpu().clone()

    pre_handle = input_layernorm.register_forward_pre_hook(_pre_hook)
    post_handle = input_layernorm.register_forward_hook(_post_hook)
    try:
        inputs_embeds = decode_inputs["inputs_embeds"].cpu()
        past_key_values = decode_inputs["past_key_values"]
        step(inputs_embeds, past_key_values)
    finally:
        pre_handle.remove()
        post_handle.remove()

    before_path = output_dir / "hidden_before_input_layernorm_layer0.pt"
    after_path = output_dir / "hidden_after_input_layernorm_layer0.pt"
    embeds_path = output_dir / "inputs_embeds_decode.pt"

    torch.save(captured["before"], before_path)
    torch.save(captured["after"], after_path)
    torch.save(decode_inputs["inputs_embeds"].detach().cpu(), embeds_path)

    manifest = {
        "variant": variant_label,
        "layer_idx": layer_idx,
        "tensors": {
            "hidden_before_input_layernorm_layer0": list(captured["before"].shape),
            "hidden_after_input_layernorm_layer0": list(captured["after"].shape),
            "inputs_embeds_decode": list(decode_inputs["inputs_embeds"].shape),
        },
        "dtype": str(captured["before"].dtype),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved layer-0 layernorm tensors under {output_dir}")
    print(f"  {before_path.name}  shape={tuple(captured['before'].shape)}")
    print(f"  {after_path.name}   shape={tuple(captured['after'].shape)}")
    print(f"  {embeds_path.name}           shape={tuple(decode_inputs['inputs_embeds'].shape)}")
    return output_dir

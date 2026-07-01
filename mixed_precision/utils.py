# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for mixed-precision heuristics: model loading, weight streaming, path helpers."""

import inspect
import json
import os

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.utils import cached_file

EXPERIMENTS_DIR = "mixed_precision_experiments"


def _resolve_model_path(model_name_or_path):
    """Resolve a HF model name or local path to a directory with safetensors files."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    try:
        config_file = cached_file(
            model_name_or_path, "config.json", local_files_only=True
        )
        return os.path.dirname(config_file)
    except Exception as e:
        raise RuntimeError(
            f"Cannot find model at '{model_name_or_path}'. "
            "Pass a local directory path or ensure the model is in the HF cache."
        ) from e


def _build_weight_map(model_path):
    """Return {tensor_name: shard_filename} for all tensors in the model."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    single = os.path.join(model_path, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _detect_weight_paths(weight_map):
    """Detect the safetensors key prefixes for the LM body and lm_head.

    Returns (lm_base, lm_head_base) where:
      lm_base      — prefix before 'embed_tokens.' / 'layers.N.' / 'norm.'
      lm_head_base — prefix before 'lm_head.'

    Examples:
      Standard decoder-only:  lm_base='model.',       lm_head_base=''
      LLaVA-style VLM:        lm_base='language_model.model.', lm_head_base='language_model.'
      Qwen3.6-27B VLM:        lm_base='model.language_model.', lm_head_base='model.language_model.'
    """
    lm_base = None
    lm_head_base = None

    for key in weight_map:
        if lm_base is None and "embed_tokens.weight" in key:
            lm_base = key[: key.index("embed_tokens.weight")]
        if lm_head_base is None and "lm_head.weight" in key:
            lm_head_base = key[: key.index("lm_head.weight")]
        if lm_base is not None and lm_head_base is not None:
            break

    return lm_base or "model.", lm_head_base or ""


def _dequantize_mxfp4_inplace(raw, device):
    """Dequantize MXFP4 *_blocks/*_scales pairs to bf16 in place.

    Some MoE models (e.g. GPT-OSS-120B) store expert weights on disk in MXFP4
    format where each weight tensor is split into a packed-integer *_blocks
    tensor and a per-block *_scales tensor. Replace them with a single
    dequantized bf16 tensor.
    """
    for full_name in list(raw):
        if not full_name.endswith("_blocks"):
            continue
        base = full_name[: -len("_blocks")]
        scales_key = base + "_scales"
        if scales_key not in raw:
            continue
        raw[base] = convert_moe_packed_tensors(
            raw.pop(full_name).to(device),
            raw.pop(scales_key).to(device),
            dtype=torch.bfloat16,
        )


def load_tensors_to_layer(layer, prefix, weight_map, model_path, device):
    """Load all tensors whose name starts with prefix from safetensors shards to device.

    Tensors go disk → device without staging the full model in CPU RAM.
    Uses accelerate's set_module_tensor_to_device to convert meta tensors.
    """
    shards = {}
    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith(prefix):
            shards.setdefault(shard_file, []).append(tensor_name)

    raw = {}
    for shard_file, tensor_names in shards.items():
        with safe_open(os.path.join(model_path, shard_file), framework="pt") as f:
            for full_name in tensor_names:
                raw[full_name] = f.get_tensor(full_name)

    _dequantize_mxfp4_inplace(raw, device)

    for full_name, tensor in raw.items():
        param_name = full_name[len(prefix) :]
        set_module_tensor_to_device(layer, param_name, device, value=tensor.to(device))


def load_model_shell(model_name_or_path, trust_remote_code=False):
    """Create an empty model shell for disk-based weight streaming.

    Returns (model, tokenizer, weight_map, model_path, lm_base, lm_head_base).
    The model has meta tensors; no weights are in RAM.

    lm_base      — safetensors key prefix before 'embed_tokens.' / 'layers.N.' / 'norm.'
    lm_head_base — safetensors key prefix before 'lm_head.'
    """
    model_path = _resolve_model_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    # VLMs wrap the LM config under text_config; use that for the causal LM shell
    lm_config = getattr(config, "text_config", config)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            lm_config, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
        )
    model.eval()

    weight_map = _build_weight_map(model_path)
    lm_base, lm_head_base = _detect_weight_paths(weight_map)
    return model, tokenizer, weight_map, model_path, lm_base, lm_head_base


def _run_layer(layer, hidden_states, position_ids, position_embeddings=None):
    seq_len = hidden_states.shape[1]
    causal_mask = torch.triu(
        torch.full(
            (seq_len, seq_len),
            float("-inf"),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        diagonal=1,
    )

    kwargs = {}
    params = inspect.signature(layer.forward).parameters
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None and "position_embeddings" in params:
        kwargs["position_embeddings"] = position_embeddings
    if "attention_mask" in params:
        kwargs["attention_mask"] = causal_mask

    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _compute_position_embeddings(base_model, seq_len, position_ids, device):
    """Compute RoPE cos/sin for models with a model-level rotary_emb."""
    if not hasattr(base_model, "rotary_emb"):
        return None
    rotary_cls = type(base_model.rotary_emb)
    rotary_emb = rotary_cls(config=base_model.config, device=device)
    dummy = torch.zeros(1, seq_len, 1, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        position_embeddings = rotary_emb(dummy, position_ids)
    del rotary_emb, dummy
    return position_embeddings


def collect_weights(model):
    """Return [(name, param)] for all quantizable weight tensors."""
    return [
        (f"{name}.{pname}" if name else pname, param)
        for name, module in model.named_modules()
        for pname, param in module.named_parameters(recurse=False)
        if param.ndim >= 2
        and not isinstance(module, nn.Embedding)
        and "norm" not in name
        and "router" not in name
        and pname != "bias"
    ]


def get_fii_path(model_name):
    model_short = model_name.split("/")[-1]
    return os.path.join(EXPERIMENTS_DIR, "fisher", model_short, "fii.pt")


def get_scores_path(model_name):
    model_short = model_name.split("/")[-1]
    return os.path.join(
        EXPERIMENTS_DIR,
        "sensitivity_scores",
        model_short,
        f"sensitivity_{model_short}.json",
    )

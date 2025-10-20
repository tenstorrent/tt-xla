# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
training_utils.py

Utility module for extracting a single `torch.Tensor` from various
model output objects (e.g., Hugging Face model outputs).

It maintains a registry of handlers that define how to unpack each output type.
Intended to be used if the model returns a class that can be unambiguously unpacked into a single tensor that represents the whole output.
In case model returns an ambiguous output (e.g, list/tuple) ModelLoader needs to override `unpack_forward_output`.
"""
import torch
from typing import Any, Callable, Dict

_HANDLER_REGISTRY: Dict[str, Callable[[Any], torch.Tensor]] = {}


def _register_attr(cls_name: str, attr: str) -> None:
    def attr_handler(x: Any) -> torch.Tensor:
        if not hasattr(x, attr):
            raise ValueError(f"Attribute {attr} does not exist in {cls_name}.")
        return getattr(x, attr)

    _register_handler(cls_name, attr_handler)


def _register_handler(cls_name: str, fn: Callable[..., torch.Tensor]) -> None:
    if cls_name in _HANDLER_REGISTRY:
        raise ValueError(f"Handler for {cls_name} already exists.")

    _HANDLER_REGISTRY[cls_name] = fn


def unpack_forward_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    cls_name = output.__class__.__name__

    if cls_name in _HANDLER_REGISTRY:
        v = _HANDLER_REGISTRY[cls_name](output)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(f"Handler for {cls_name} did not return a torch.Tensor")

    raise ValueError(
        f"No handler for class {cls_name} exists in `unpack_forward_output`."
        f"Register a handler or implement custom unpack_forward_output for the specific model."
    )


_register_attr("BaseModelOutputWithPast", "last_hidden_state")
_register_attr("BaseModelOutputWithPastAndCrossAttentions", "last_hidden_state")
_register_attr("CausalLMOutputWithCrossAttentions", "logits")
_register_attr("CausalLMOutputWithPast", "logits")
_register_attr("CLIPOutput", "logits_per_text")
_register_attr("DepthEstimatorOutput", "predicted_depth")
_register_attr("DPRReaderOutput", "end_logits")
_register_attr("ImageClassifierOutput", "logits")
_register_attr("ImageClassifierOutputWithNoAttention", "logits")
_register_attr("LlavaCausalLMOutputWithPast", "logits")
_register_attr("MambaCausalLMOutput", "logits")
_register_attr("MaskedLMOutput", "logits")
_register_attr("MgpstrModelOutput", "logits")
_register_attr("PerceiverClassifierOutput", "logits")
_register_attr("PerceiverMaskedLMOutput", "logits")
_register_attr("SegFormerImageClassifierOutput", "logits")
_register_attr("Seq2SeqLMOutput", "logits")
_register_attr("Seq2SeqSequenceClassifierOutput", "logits")
_register_attr("SequenceClassifierOutput", "logits")
_register_attr("SequenceClassifierOutputWithPast", "logits")
_register_attr("TokenClassifierOutput", "logits")
_register_attr("UNet2DConditionOutput", "sample")

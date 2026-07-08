# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import OrderedDict

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from .layers.mm_embeddings import install_static_shape_merge_multimodal_embeddings
from .layers.mrope import override_mrope_module
from .layers.multimodal_attention import override_vision_attention
from .layers.rmsnorm import override_rmsnorm_module
from .layers.rotary_embedding import override_rotary_embedding_module
from .logger import tt_init_logger

logger = tt_init_logger(__name__)

# Patch vLLM's multimodal embedding merge to a static-shape impl at import time.
install_static_shape_merge_multimodal_embeddings()


def get_fqn(module):
    return module.__class__.__qualname__


MODULE_TYPE_TO_TT_OVERRIDE = OrderedDict(
    [
        ("RMSNorm", override_rmsnorm_module),
    ]
)

# isinstance-based overrides for classes where subclasses need the same treatment
ISINSTANCE_OVERRIDES = [
    (MRotaryEmbedding, override_mrope_module),
    (RotaryEmbedding, override_rotary_embedding_module),
]


def _promote_pre_allocated_attrs_to_buffers(model: torch.nn.Module) -> None:
    """Re-register plain torch.Tensor attributes as buffers so .to() moves them.

    Some multimodal models pre-allocate ``self.per_layer_embeddings`` as a
    plain attribute (not a registered buffer) on CPU. ``model.to(device)``
    only relocates parameters and registered buffers, leaving it stranded on
    CPU; a later add against an XLA tensor then trips dynamo's mixed-device
    check. Promote such attributes to non-persistent buffers so .to() follows
    them.
    """
    pre_allocated_attrs = ("per_layer_embeddings",)
    for attr in pre_allocated_attrs:
        if not hasattr(model, attr):
            continue
        t = getattr(model, attr)
        if not isinstance(t, torch.Tensor):
            continue
        if attr in dict(model.named_buffers(recurse=False)):
            continue
        delattr(model, attr)
        model.register_buffer(attr, t, persistent=False)


def replace_modules(model: torch.nn.Module) -> None:
    logger.info(
        "Replacing vLLM modules with TT-compatible overrides where necessary..."
    )

    def _find_override(module):
        fqn = get_fqn(module)
        if fqn in MODULE_TYPE_TO_TT_OVERRIDE:
            return MODULE_TYPE_TO_TT_OVERRIDE[fqn](module)
        for base_cls, override_fn in ISINSTANCE_OVERRIDES:
            if isinstance(module, base_cls):
                return override_fn(module)
        return None

    def _process_module(module, name=None, parent=None):
        replacement = _find_override(module)
        if replacement is not None:
            assert (
                parent is not None and name is not None
            ), "Top Level module is not expected to be wrapped."
            logger.debug("replace %s with %s", module, replacement)
            setattr(parent, name, replacement)
            module = replacement

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
    override_vision_attention(model)
    _promote_pre_allocated_attrs_to_buffers(model)


def repair_stale_moe_closures(model: torch.nn.Module) -> None:
    """Move CPU tensors captured by MoE ``custom_routing_function`` closures
    onto the model's device.

    Models that build ``custom_routing_function`` as a closure over a parameter
    (e.g. Gemma-4's ``per_expert_scale``) keep the original CPU tensor in the
    closure cell after ``model.to(device)``, so routing later mixes cpu/xla
    tensors and fails to trace. Rewrite any such cell to the device tensor —
    name- and model-agnostic. Runs unconditionally after load, independent of TP.
    """
    device = next((p.device for p in model.parameters()), None)
    if device is None or device.type == "cpu":
        return

    for module in model.modules():
        fn = getattr(module, "custom_routing_function", None)
        closure = getattr(fn, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue  # empty cell
            if isinstance(val, torch.Tensor) and val.device != device:
                cell.cell_contents = val.to(device)

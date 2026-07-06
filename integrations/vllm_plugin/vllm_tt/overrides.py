# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import OrderedDict

import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from .layers.mrope import override_mrope_module
from .layers.rmsnorm import override_rmsnorm_module
from .layers.rotary_embedding import override_rotary_embedding_module
from .logger import tt_init_logger

logger = tt_init_logger(__name__)


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

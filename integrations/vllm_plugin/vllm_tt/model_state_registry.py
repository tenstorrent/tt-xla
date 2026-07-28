# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model architecture -> TT ``ModelState`` dispatch for MRv2.

Upstream models expose ``get_model_state_cls()``, but it returns the GPU
``ModelState`` (Triton-backed, does not run on TT), so the v2 runner cannot use
it. Map architectures to their TT ``ModelState`` here instead; anything
unmapped gets the standard ``TTModelState``.

Entries are ``"module:attr"`` strings resolved on first use, so registering a
state class never drags its imports into module load.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from .model_state import TTModelState

_REGISTRY: dict[str, str] = {
    "DiffusionGemmaForConditionalGeneration": (
        "vllm_tt.diffusion_gemma:TTDiffusionGemmaModelState"
    ),
}


def register_tt_model_state(arch: str, target: str) -> None:
    """Register ``arch`` (a model class name) -> ``"module:attr"``."""
    _REGISTRY[arch] = target


def _resolve(target: str) -> type:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def get_tt_model_state_cls(model: "nn.Module") -> type["TTModelState"]:
    """The TT ``ModelState`` class backing ``model``.

    Walks the MRO so a subclass of a registered architecture still matches.
    """
    from .model_state import TTModelState

    for cls in type(model).__mro__:
        target = _REGISTRY.get(cls.__name__)
        if target is not None:
            return _resolve(target)
    return TTModelState

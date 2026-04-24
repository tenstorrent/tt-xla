# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy compatibility shim for `tt_torch.sparse_mlp`.

The full sparse-MoE module has been replaced by `tt_torch.moe_backend` (the
HF `ExpertsInterface`-based `tt_moe` backend). The two symbols below exist
only so that the out-of-tree vendored Kimi K2 loader under
`third_party/tt_forge_models/kimi_k2/pytorch/loader.py` — which is checked in
as a git submodule and can't be updated atomically with this change — keeps
importing.

`enable_sparse_mlp` now installs the `tt_moe` backend globally and
propagates the `_experts_implementation` flag to the model's config so the
HF `@use_experts_implementation` decorator routes Experts.forward at call
time. It is a no-op on Experts classes that don't follow the HF canonical
layout (e.g. Kimi K2's vendored `DeepseekV3MoE` with `nn.ModuleList` of
experts) — those paths fall back to HF eager.

`A2aSparseMLPWithSharedExperts` is kept only so that legacy `isinstance`
checks evaluate to `False` rather than raising; nothing in-tree instantiates
it any more.

Once the kimi_k2 submodule is refreshed to use `tt_torch.moe_backend`
directly this file should be deleted.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .moe_backend import TT_MOE_BACKEND_NAME, register_tt_moe_backend


class A2aSparseMLPWithSharedExperts(nn.Module):
    """Placeholder for the legacy type; never instantiated in-tree now."""


def enable_sparse_mlp(model: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    """Legacy entry point.

    Back-compat shim that registers the `tt_moe` backend and marks the
    model's config so its HF-canonical Experts modules dispatch through it.
    Returns ``model`` unchanged (no wrapper replacement).
    """
    cluster_axis = int(kwargs.get("cluster_axis", 0))
    register_tt_moe_backend(cluster_axis=cluster_axis)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg._experts_implementation = TT_MOE_BACKEND_NAME
    for sub in model.modules():
        sub_cfg = getattr(sub, "config", None)
        if sub_cfg is not None:
            sub_cfg._experts_implementation = TT_MOE_BACKEND_NAME

    return model

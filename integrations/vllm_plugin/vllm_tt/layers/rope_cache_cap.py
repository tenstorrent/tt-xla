# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cap the DeepSeek-V4 (deepseek_yarn) RoPE cos/sin cache length on TT.

DSV4-Flash ships ``rope_scaling`` YaRN with ``factor=16`` and
``original_max_position_embeddings=65536``. vLLM's ``get_rope`` builds the
``deepseek_yarn`` cos/sin cache with ``arange(original_max * factor)`` =
``arange(1_048_576)`` (see ``DeepseekV4ScalingRotaryEmbedding.
_compute_cos_sin_cache``). At ``rope_dim=64`` that is a ~268 MB fp32 buffer;
tilizing/loading it onto a Tenstorrent mesh is impractical (and was the trigger
that first wedged the device during DSV4 bring-up).

The cache is just ``cos(t * inv_freq)`` / ``sin(t * inv_freq)`` for
``t = 0 .. len-1``. The frequencies (``_compute_inv_freq``) do **not** depend on
the cache length, so truncating ``t`` to the largest position the engine can
actually index (``max_model_len``) yields a cache whose every row is
bit-identical to the full one — RoPE stays exact, memory drops to KBs.
Indexing a position ``>= cap`` would raise (loud), not silently corrupt.

``install()`` monkeypatches ``_compute_cos_sin_cache`` on both the base
``DeepseekScalingRotaryEmbedding`` and the ``DeepseekV4ScalingRotaryEmbedding``
subclass. The cap is ``TT_ROPE_CACHE_CAP`` (env, if set) else the current
vLLM config's ``max_model_len``; if neither is available it falls back to a
bounded default rather than materializing the full table.
"""
from __future__ import annotations

import os

import torch

from ..logger import tt_init_logger

logger = tt_init_logger(__name__)

# Bounded fallback when neither the env override nor a readable vLLM config is
# available (should not happen during normal model init, which runs inside
# set_current_vllm_config). 131072 rows @ rope_dim*2 fp32 is a few tens of MB —
# safe, and large enough for typical bring-up sequence lengths.
_DEFAULT_CAP = 131072


def _resolve_cap() -> int:
    env = os.environ.get("TT_ROPE_CACHE_CAP")
    if env:
        return int(env)
    try:
        from vllm.config import get_current_vllm_config

        mml = get_current_vllm_config().model_config.max_model_len
        if mml and mml > 0:
            return int(mml)
    except Exception:
        pass
    logger.warning(
        "RoPE cache cap: no TT_ROPE_CACHE_CAP and no readable max_model_len; "
        "falling back to %d rows.",
        _DEFAULT_CAP,
    )
    return _DEFAULT_CAP


def _capped_compute_cos_sin_cache(self) -> torch.Tensor:
    inv_freq = self._compute_inv_freq(self.scaling_factor)
    full_len = int(self.max_position_embeddings * self.scaling_factor)
    cap = _resolve_cap()
    n = min(full_len, cap)
    if n < full_len:
        logger.info(
            "Capping DSV4 RoPE cos/sin cache: %d -> %d rows "
            "(exact freqs preserved; positions >= %d unreachable at "
            "max_model_len).",
            full_len,
            n,
            n,
        )
    # Build on CPU fp32 (matches base-class dtype); the buffer is moved to the
    # device with the model. Avoids a device-side 1M arange during init.
    t = torch.arange(n, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos() * self.mscale
    sin = freqs.sin() * self.mscale
    return torch.cat((cos, sin), dim=-1)


_INSTALLED = False


def install() -> None:
    """Monkeypatch the deepseek YaRN rope classes to cap the cache length."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
        DeepseekV4ScalingRotaryEmbedding,
    )

    DeepseekScalingRotaryEmbedding._compute_cos_sin_cache = (
        _capped_compute_cos_sin_cache
    )
    DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache = (
        _capped_compute_cos_sin_cache
    )
    _INSTALLED = True
    logger.info("Installed DSV4 RoPE cos/sin cache cap.")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

import torch

from ..logger import tt_init_logger

logger = tt_init_logger(__name__)


def install_static_shape_merge_multimodal_embeddings() -> None:
    """Replace vLLM's masked_scatter_-based _merge_multimodal_embeddings: the
    boolean-mask scatter lowers to a dynamic-dim (set_dimension_size) tensor
    that tt-mlir's Shardy pass rejects. cumsum+gather+where gives the same
    result with static shapes. Idempotent (attribute-flag guarded)."""
    import vllm.model_executor.models.utils as _vllm_utils

    if getattr(_vllm_utils, "_tt_static_shape_merge_installed", False):
        return

    def _tt_merge_multimodal_embeddings(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        if len(multimodal_embeddings) == 0:
            return inputs_embeds
        mm_embeds_flat = _vllm_utils._flatten_embeddings(multimodal_embeddings).to(
            dtype=inputs_embeds.dtype
        )
        # Zero-based positional index of each mm token among the mm tokens
        # (0, 0, 1, 1, 2, ... where ascents happen at mm positions). Subtract 1
        # so non-mm positions point to index -1 (clamped to 0 below for safety).
        mm_indices = is_multimodal.to(torch.int64).cumsum(dim=0) - 1
        mm_indices = mm_indices.clamp(min=0)
        # Gather mm embeddings at those indices for every position. Non-mm
        # positions read garbage which torch.where then discards.
        mm_padded = mm_embeds_flat[mm_indices]
        merged = torch.where(is_multimodal.unsqueeze(-1), mm_padded, inputs_embeds)
        return merged

    _vllm_utils._merge_multimodal_embeddings = _tt_merge_multimodal_embeddings
    _vllm_utils._tt_static_shape_merge_installed = True
    logger.info(
        "Installed static-shape _merge_multimodal_embeddings (replaces "
        "masked_scatter_-based default that emits dynamic shapes)."
    )

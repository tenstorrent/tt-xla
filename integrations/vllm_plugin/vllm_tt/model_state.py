# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT ``ModelState`` for the vLLM v2 model runner.

``DefaultModelState`` is not subclassed: its ``__init__`` builds Triton-backed
rope state and its ``prepare_attn`` is cudagraph-oriented, so a standalone
implementation is needed to substitute TT equivalents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.tasks import GenerationTask
from vllm.v1.worker.gpu.model_states.interface import ModelState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.v1.worker.gpu.states import RequestState


class TTModelState(ModelState):
    """Tenstorrent implementation of the ``ModelState`` interface."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        model: nn.Module,
        encoder_cache: "EncoderCache | None",
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.supports_mm_inputs = encoder_cache is not None
        self.encoder_cache = encoder_cache
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dtype = self.model_config.dtype
        # Set here because the base __init__ is skipped (it builds a CUDA
        # EncoderRunner), but inherited hooks still read this.
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()

        # No TT rope substitute yet; the text path uses plain 1D positions.
        # TODO(mrv2): TT rope/mrope state for models with uses_mrope.
        self.rope_state = None

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        """Which generation tasks this model supports."""
        from vllm.model_executor.models.interfaces import (
            supports_realtime,
            supports_transcription,
        )
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        supported_tasks: list[GenerationTask] = []

        if is_text_generation_model(self.model):
            supported_tasks.append("generate")

        if supports_transcription(self.model):
            if self.model.supports_transcription_only:
                return ("transcription",)
            supported_tasks.append("transcription")

        if supports_realtime(self.model):
            supported_tasks.append("realtime")

        return tuple(supported_tasks)

    def add_request(self, req_index: int, new_req_data: "NewRequestData") -> None:
        """Per-request model-specific setup. No-op without rope state."""
        if self.rope_state is not None:
            # TODO(mrv2): init TT rope prefill positions for req_index.
            raise NotImplementedError("TT rope state not implemented yet.")

    def apply_staged_writes(self) -> None:
        """Commit staged host->device writes. No-op without rope state."""
        if self.rope_state is not None:
            raise NotImplementedError("TT rope state not implemented yet.")

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        """Post-step model-specific state update. No-op for standard models."""
        return None

    def prepare_inputs(
        self, input_batch: "InputBatch", req_states: "RequestState"
    ) -> dict[str, Any]:
        """Extra kwargs merged into the model call.

        Empty on the 1D-positions path, so the runner's own input_ids/positions/
        inputs_embeds are used unchanged.
        """
        if self.rope_state is None:
            return {}
        # TODO(mrv2): compute TT mrope positions and return {"positions": ...}.
        raise NotImplementedError("TT rope state not implemented yet.")

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """Kwargs for dummy/warmup/profile runs."""
        return {}

    def get_mm_embeddings(
        self,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
        mm_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Merge text + gathered multimodal embeddings.

        Scatters with a static ``index_copy``, not ``masked_scatter_``, which
        lowers to a dynamic-shaped op the Shardy/SPMD pass rejects.
        """
        inputs_embeds = self.model.embed_input_ids(input_ids, is_multimodal=is_mm_embed)
        if mm_embeds:
            mm_flat = torch.cat(list(mm_embeds)).to(inputs_embeds.dtype)
            original_shape = inputs_embeds.shape
            hidden_size = original_shape[-1]
            inputs_embeds = inputs_embeds.reshape(-1, hidden_size)
            inputs_embeds = inputs_embeds.index_copy(0, mm_indices, mm_flat)
            inputs_embeds = inputs_embeds.reshape(original_shape)
        return inputs_embeds

    def prepare_attn(
        self,
        attention_layer_names: "Iterable[str]",
        page_table: torch.Tensor,
        cache_position: torch.Tensor,
        fill_page_table: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        num_users: int | None = None,
        dp_size: int = 1,
        chunk_start_idx: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> dict[str, Any]:
        """Fan one TTMetadata out to every attention layer.

        The runner computes the paged tensors host-side. fill_page_table defaults
        to page_table (no prefix roll).
        """
        from .attention_impls.attention import TTMetadata

        attn_metadata = TTMetadata(
            page_table=page_table,
            cache_position=cache_position,
            is_causal=is_causal,
            attn_mask=attn_mask,
            fill_page_table=fill_page_table,
            dp_size=dp_size,
            chunk_start_idx=chunk_start_idx,
            batch_idx=batch_idx,
            num_users=num_users,
        )
        return dict.fromkeys(attention_layer_names, attn_metadata)

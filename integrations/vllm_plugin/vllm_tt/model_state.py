"""TT ``ModelState`` for vLLM Model Runner v2 (MRv2).

This is Phase 1 of the MRv2 adoption: the model-specific ``ModelState`` layer.
It implements the upstream ``vllm.v1.worker.gpu.model_states.interface.ModelState``
contract (as of vLLM v0.22.1) for Tenstorrent hardware.

Scope / status
--------------
MRv2 splits the old monolithic runner into four pieces: the runner (engine),
``ModelState`` (model-specific logic), ``RequestState`` (persistent per-request
table) and ``InputBatch`` (transient per-step view). This file is only the
``ModelState`` piece.

* Requires vLLM v0.22.1. The ``vllm.v1.worker.gpu.*`` modules imported below do
  not exist on v0.19.1, so this module is intentionally NOT imported from the
  package ``__init__`` yet -- importing it on an un-uplifted env raises
  ImportError. It becomes live once the 0.22.1 uplift (PR #5634) lands and the
  TT v2 runner (Phase 3) wires it in.
* The model-agnostic methods (task discovery, the common 1D-positions input
  path, staged-write / add_request no-ops) are implemented for real here.
* The two methods that are coupled to the runner and to TT's own attention
  backends -- ``prepare_attn`` and ``get_mm_embeddings`` -- raise
  NotImplementedError with the exact contract documented. They are only
  meaningfully implementable and testable once the TT v2 runner exists
  (Phase 3), so we do not ship a fake body here.

Why not subclass upstream ``DefaultModelState``?
------------------------------------------------
``DefaultModelState.__init__`` builds rope state via ``get_rope_state`` (which
is Triton-backed, ``@triton.jit``) and ``prepare_attn`` delegates to
``build_attn_metadata`` (cudagraph-oriented). Both assume CUDA/Triton, so
subclassing would drag that machinery in at construction time. A standalone
implementation lets us substitute TT equivalents cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
from vllm.v1.worker.gpu.model_states.interface import ModelState

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


class TTModelState(ModelState):
    """Tenstorrent implementation of the MRv2 ``ModelState`` interface.

    Mirrors the structure of upstream ``DefaultModelState`` but substitutes (or
    defers) the CUDA/Triton-specific pieces. See module docstring for status.
    """

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

        # Sizing, mirrored from DefaultModelState.
        self.supports_mm_inputs = encoder_cache is not None
        self.encoder_cache = encoder_cache
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dtype = self.model_config.dtype

        # Rope/mrope state: upstream uses get_rope_state(), which is Triton-backed
        # and does not run on TT. The common text-generation path uses plain 1D
        # positions (rope_state is None -> prepare_inputs returns {}), matching
        # DefaultModelState's common case. mrope models require a TT rope
        # substitute; deferred to a later phase.
        # TODO(mrv2): TT rope/mrope state for models with uses_mrope.
        self.rope_state = None

        # TODO(mrv2): construct the TT multimodal encoder runner here when
        # get_mm_embeddings is implemented (Phase 3). Upstream builds an
        # EncoderRunner; portability to TT is not yet validated.

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        """Which generation tasks this model supports.

        Fully portable: same logic as upstream DefaultModelState and the current
        TT v1 runner. Lazy imports mirror upstream to avoid import cycles.
        """
        from vllm.model_executor.models.interfaces import (
            supports_realtime,
            supports_transcription,
        )
        from vllm.model_executor.models.interfaces_base import (
            is_text_generation_model,
        )

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
        """Per-request model-specific setup.

        Upstream initializes rope prefill positions here. With no TT rope state
        yet (common text path), this is a no-op, matching DefaultModelState when
        rope_state is None.
        """
        if self.rope_state is not None:
            # TODO(mrv2): init TT rope prefill positions for req_index.
            raise NotImplementedError("TT rope state not implemented yet.")

    def apply_staged_writes(self) -> None:
        """Commit any staged host->device writes owned by this ModelState.

        Only rope state stages writes upstream; no-op until TT rope lands.
        """
        if self.rope_state is not None:
            raise NotImplementedError("TT rope state not implemented yet.")

    def postprocess_state(
        self, input_batch: "InputBatch", num_sampled: torch.Tensor
    ) -> None:
        """Post-step model-specific state update. No-op for standard models."""
        return None

    def prepare_inputs(
        self, input_batch: "InputBatch", req_states: "RequestState"
    ) -> dict[str, Any]:
        """Extra kwargs merged into the model call.

        Common case (1D positions, no rope state): return {} so the runner's
        default input_ids/positions/inputs_embeds are used unchanged. This
        matches DefaultModelState.prepare_inputs. mrope models return
        {"positions": ...}; deferred with TT rope.
        """
        if self.rope_state is None:
            return {}
        # TODO(mrv2): compute TT mrope positions and return {"positions": ...}.
        raise NotImplementedError("TT rope state not implemented yet.")

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """Kwargs for dummy/warmup/profile runs.

        Common text path needs no extra inputs. mm/rope models add
        inputs_embeds/positions; deferred with those substitutes.
        """
        return {}

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: "InputBatch",
        req_states: "RequestState",
    ) -> torch.Tensor | None:
        """Run the multimodal encoder and merge embeddings into input embeds.

        Contract (from DefaultModelState.get_mm_embeddings):
          1. prepare_mm_inputs(scheduled_encoder_inputs) -> (hashes, kwargs)
          2. if kwargs: execute the encoder, cache outputs by mm_hash
          3. gather per-token mm embeds and merge with text embeds
          4. return inputs_embeds[:input_batch.num_tokens_after_padding]

        Deferred to Phase 3: needs the TT multimodal encoder runner and the
        runner-owned InputBatch. Salvage source in the current v1 runner:
        model_runner.py::_execute_mm_encoder and ::_gather_mm_embeddings.
        """
        raise NotImplementedError(
            "get_mm_embeddings requires the TT v2 runner + TT encoder runner "
            "(MRv2 Phase 3)."
        )

    def prepare_attn(
        self,
        input_batch: "InputBatch",
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: "list[list[AttentionGroup]]",
        kv_cache_config: "KVCacheConfig",
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """Build the attention-metadata dict passed to the model forward.

        Upstream DefaultModelState delegates to build_attn_metadata(), which is
        cudagraph-oriented and drives per-backend metadata builders. TT must
        build metadata for its OWN attention backends (see attention_impls/),
        so this is not a straight reuse.

        Deferred to Phase 3: the physical block_tables/slot_mappings are produced
        by the runner's own prepare_attn, which does not exist for TT yet.
        Salvage source in the current v1 runner:
        model_runner.py::_get_slot_mapping_metadata and the attn-metadata build.
        """
        raise NotImplementedError(
            "prepare_attn requires the TT v2 runner + TT attention metadata "
            "build (MRv2 Phase 3)."
        )

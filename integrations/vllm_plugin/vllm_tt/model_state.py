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
* ``prepare_attn`` (Phase 3) assembles a ``TTMetadata`` from the per-step
  device arrays the runner computes host-side and fans it out to every attention
  layer -- the same ``dict.fromkeys(layer_names, meta)`` the v1 fork builds
  inline. Its signature diverges from upstream's (which passes flat
  ``block_tables``/``slot_mappings``): TT's paged ops consume
  ``page_table``/``cache_position`` instead, so matching upstream would mean
  re-porting Triton block-table kernels TT never uses.
* ``get_mm_embeddings`` still raises NotImplementedError: its data seams (the
  per-request multimodal-feature store, the padded ``input_ids`` layout it masks
  against, and ``model.embed_multimodal``/``embed_input_ids``) are owned by the
  TT v2 runner, which does not exist yet. It is co-implemented with that runner
  rather than against a speculative interface. Contract documented on the method.

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
        """Assemble the per-layer attention-metadata dict for the model forward.

        Upstream DefaultModelState delegates to build_attn_metadata() over flat
        block_tables/slot_mappings; TT's attention backends (see attention_impls/)
        consume a single ``TTMetadata`` built from paged tensors instead, so the
        signature is adapted (see module docstring). The per-step tensors are
        computed host-side by the runner; this method only packages them and fans
        the shared metadata out to every attention layer -- mirroring the v1
        fork's ``dict.fromkeys(self._attention_layer_names, attn_metadata)``.
        ``fill_page_table`` defaults to ``page_table`` (no prefix roll).
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

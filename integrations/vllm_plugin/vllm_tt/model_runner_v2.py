# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT Model Runner v2 (MRv2) -- fork of upstream ``vllm/v1/worker/gpu/model_runner.py``.

Phase 3 of the MRv2 adoption. Upstream's v2 runner is Triton/CUDA/UVA-native
throughout, so TT forks it and substitutes host-side mechanisms rather than
subclassing ``GPUModelRunner``.

This file is built in reviewable sub-increments. Present so far: the request
lifecycle, the decode-first batch selection / SMEM multi-pass clamping, the host
input-token preparation, the attention slot-mapping build (feeding
``TTModelState.prepare_attn``), the sampled-token writeback (``postprocess``),
the ``execute_model``/``sample_tokens`` two-phase driver that threads them
together, ``initialize_kv_cache`` (device KV allocation), and
``__init__``/``load_model`` (single-device state construction + model load /
compile), and ``_run_model_pass`` + the compiled forward/sample graph. The
runner is structurally complete for the single-device greedy text path, with the
worker-facing surface (``get_kv_cache_spec``, ``capture_model``, ``get_model``,
``reset_mm_cache``) in place. Deferred: the multi-device SPMD/mesh path out of
``__init__``, ``get_mm_embeddings``, LoRA (``add_lora``), and the grammar /
cpu_sampling / prompt-logprobs branches (driver-gated).

Layout note
-----------
TT's model forward is **2D** ``[num_reqs, padded_query_len]`` (optionally
reshaped to 1D at the call boundary for ``flat_model_io`` models), unlike
upstream's flat ``[num_tokens]`` layout. So the runner builds 2D
``input_ids``/``positions`` itself (as the v1 fork does); ``TTInputBatch`` (a
verbatim upstream port, flat) serves as the per-step bookkeeping view
(``idx_mapping``/``num_scheduled_tokens``/``query_start_loc``/``seq_lens``/
``logits_indices``) consumed by ``from_v2_states`` and the attn build.

Lifecycle scope
---------------
``add_requests``/``update_requests``/``finish_requests`` drive the split v2
state, following upstream v2 semantics (NOT the v1 fork's ``_update_states``):

* Requests own a **stable slot** for their lifetime (``TTRequestState`` free
  list) -- there is no v1-style condense/shuffle.
* Preempted requests are removed here (slot freed); when the scheduler resumes
  them it re-sends them via ``scheduled_new_reqs`` with ``num_computed_tokens``
  already advanced, so there is no resumed-request branch in ``update_requests``.
* Sampling params go to ``TTSamplingStates`` (keyed by the same slot), token
  bookkeeping to ``TTRequestState``, block ids to the runner-owned block table.

The runner instance is expected to carry (set up by ``__init__``, later
sub-increment): the split state (``req_states``, ``sampling_states``,
``block_table`` -- a vLLM ``MultiGroupBlockTable``, ``encoder_cache`` dict,
``num_prompt_logprobs`` dict, ``model_state``); scalars (``vocab_size``,
``max_num_blocks_per_req``, ``supports_mm_inputs``, and the SMEM-cap set read by
``_select_batch``: ``most_model_len``/``num_reqs_max_model_len``/
``num_reqs_most_model_len``/``min_num_reqs``/``max_prefill_num_reqs``/
``max_num_reqs``/``num_tokens_paddings``); the per-step handoff
``scheduler_output`` (initialised to None); and the device machinery reached
only through ``_run_model_pass`` (persistent input/attn buffers, ``model``,
``sampler``, ``sampling_device``, ``attention_layer_names``, ``dp_size``, and the
compiled forward graphs).
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.sampling_params import SamplingType
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """First padding >= x (the per-step query-length bucket)."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _adjust_min_token(min_token_size: int) -> int:
    """Round min_token_size up to a power of two that is >= 32 (32B alignment)."""
    if (min_token_size & (min_token_size - 1)) == 0 and min_token_size >= 32:
        return min_token_size
    if min_token_size > 32:
        return 1 << (min_token_size - 1).bit_length()
    return 32


def _get_token_paddings(min_token_size: int, max_token_size: int) -> list[int]:
    """Exponential token-length ladder from min_token_size up past max_token_size.

    Always starts with 1 to support single-token decode steps.
    """
    num = _adjust_min_token(min_token_size)
    paddings = [1]
    while True:
        paddings.append(num)
        if num >= max_token_size:
            break
        num *= 2
    return paddings


class TTModelRunnerV2:
    """Tenstorrent MRv2 model runner (see module docstring)."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device) -> None:
        """Construct the split v2 state + config scalars (single-device path).

        Multi-device SPMD (mesh / tensor + data parallel) is deferred; this
        constructor raises if those TT config flags are set. The model itself is
        loaded and compiled in ``load_model``; ``model``/``model_state``/``sampler``
        are None until then.
        """
        from vllm.v1.worker.block_table import MultiGroupBlockTable

        from .attention_impls.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE,
            TTAttentionBackend,
        )
        from .input_batch_v2 import TTInputBuffers
        from .platform import TTConfig
        from .request_state import TTRequestState
        from .sampling_state_v2 import TTSamplingStates

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config
        self.lora_config = vllm_config.lora_config
        self.device = device

        # additional_config arrives as a plain dict; build the typed TTConfig
        # (as the v1 runner does) so the field reads below see real values.
        self.tt_config = TTConfig(**vllm_config.additional_config)
        if self.device.type == "xla":
            import torch_xla

            torch_xla.set_custom_compile_options(
                self.tt_config.get_pjrt_compile_config()
            )

        tt = self.tt_config
        self.enable_tensor_parallel = bool(getattr(tt, "enable_tensor_parallel", False))
        self.enable_data_parallel = bool(getattr(tt, "enable_data_parallel", False))
        if self.enable_tensor_parallel or self.enable_data_parallel:
            raise NotImplementedError(
                "Multi-device (SPMD mesh) __init__ is deferred; single device only."
            )
        self.dp_size = 1
        self.mesh = None
        self.original_parallel_config = None

        self.dtype = self.model_config.dtype
        if self.cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = (
                TPU_STR_DTYPE_TO_TORCH_DTYPE[self.dtype]
                if isinstance(self.dtype, str)
                else self.dtype
            )
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]
        # 1-byte accounting stand-in so vLLM budgets blocks for the BFP8 footprint.
        self.kv_cache_spec_dtype = (
            torch.uint8
            if getattr(tt, "experimental_kv_cache_dtype", None) == "bfp_bf8"
            else self.kv_cache_dtype
        )

        self.block_size = self.cache_config.block_size
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.num_blocks_per_most_len_req = (
            cdiv(self.most_model_len, self.block_size)
            if self.most_model_len is not None
            else None
        )

        # Prefill request-count bucketing bounds (decode always uses max_num_reqs).
        self.min_num_reqs = getattr(tt, "min_num_seqs", None) or self.max_num_reqs
        self.max_prefill_num_reqs = (
            getattr(tt, "max_prefill_num_seqs", None) or self.max_num_reqs
        )

        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        min_context_len = getattr(tt, "min_context_len", 32)
        if max_num_batched_tokens < min_context_len:
            min_context_len = max_num_batched_tokens
        self.prefill_chunk_budget = min(
            getattr(
                self.scheduler_config, "tt_prefill_chunk_size", max_num_batched_tokens
            )
            or max_num_batched_tokens,
            self.max_model_len,
        )
        self.num_tokens_paddings = _get_token_paddings(
            min_context_len, self.prefill_chunk_budget
        )
        if getattr(tt, "decode_only", False):
            self.num_tokens_paddings = [1]
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model dims.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.vocab_size = self.model_config.get_vocab_size()
        self.supports_mm_inputs = bool(
            getattr(self.model_config, "is_multimodal_model", False)
        )
        self.mm_budget = None  # set when multimodal profiling lands

        # SMEM row caps consumed by _select_batch.
        self.num_reqs_max_model_len = min(
            TTAttentionBackend.get_max_num_seqs(self.max_model_len, self.block_size),
            self.max_num_reqs,
        )
        self.num_reqs_most_model_len = (
            min(
                TTAttentionBackend.get_max_num_seqs(
                    self.most_model_len, self.block_size
                ),
                self.max_num_reqs,
            )
            if self.most_model_len is not None
            else None
        )

        # Driver / graph knobs read later.
        self.use_flat_model_io = bool(getattr(tt, "flat_model_io", False))
        self.cpu_sampling = bool(getattr(tt, "cpu_sampling", False))
        self.enable_decode_fused_graphs = bool(
            getattr(tt, "enable_decode_fused_graphs", False)
        )
        self.sampling_device = torch.device("cpu") if self.cpu_sampling else self.device

        # Split v2 state.
        self.req_states = TTRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=0,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.sampling_states = TTSamplingStates(
            max_num_reqs=self.max_num_reqs, vocab_size=self.vocab_size
        )
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.block_size],
        )
        self.input_buffers = TTInputBuffers(
            self.max_num_reqs, self.max_num_tokens, self.device
        )

        self.encoder_cache: dict = {}
        self.num_prompt_logprobs: dict = {}
        self.kv_caches: list = []
        self.kv_cache_config = None
        # layer_name -> target layer for cross-layer KV sharing (get_kv_cache_spec).
        self.shared_kv_cache_layers: dict = {}
        # Per-step handoff between the two-phase execute_model / sample_tokens.
        self.scheduler_output = None

        # Filled by load_model.
        self.model = None
        self.model_state = None
        self.sampler = None
        self._attention_layer_names: tuple = ()
        self.attention_layer_names: tuple = ()

    def load_model(self) -> None:
        """Load, place, and compile the model; build ModelState + sampler.

        Single-device path (salvaged from the v1 fork): the multi-device weight
        sharding, the vocab-embedding TP-rank patch, LoRA, the num-layers override,
        and per-tensor weight-dtype overrides are deferred with the SPMD ``__init__``.
        Needs a real model + loader, so it is validated at engine stand-up.
        """
        from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.model_loader import get_model_loader

        from .model_state import TTModelState
        from .overrides import repair_stale_moe_closures, replace_modules
        from .sampler import Sampler

        loader = get_model_loader(self.load_config)
        with set_current_vllm_config(self.vllm_config):
            model = loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            ).eval()
        replace_modules(model)
        self.model = model.to(self.device)

        # Repair MoE routing closures that captured CPU tensors before to(device).
        repair_stale_moe_closures(self.model)

        self.model.compile(backend="tt", dynamic=False)
        self.sampler = Sampler()

        encoder_cache = self.encoder_cache if self.supports_mm_inputs else None
        self.model_state = TTModelState(
            self.vllm_config, self.model, encoder_cache, self.device
        )

        # Cache the attention layer names for the per-step prepare_attn fan-out.
        self._attention_layer_names = tuple(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        self.attention_layer_names = self._attention_layer_names

    def _remove_request(self, req_id: str) -> None:
        """Free a request's slot across every per-slot table. Idempotent."""
        slot = self.req_states.req_id_to_index.get(req_id)
        if slot is None:
            return
        self.req_states.remove_request(req_id)
        self.sampling_states.remove_request(slot)
        self.block_table.clear_row(slot)
        self.num_prompt_logprobs.pop(req_id, None)

    def finish_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Remove finished and preempted requests, freeing their slots.

        Preempted requests are dropped (not kept at their slot as in v1): the
        scheduler resends them through ``scheduled_new_reqs`` when resumed.
        """
        for req_id in scheduler_output.finished_req_ids:
            self._remove_request(req_id)
        for req_id in scheduler_output.preempted_req_ids:
            self._remove_request(req_id)
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

    def add_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Register newly scheduled requests into the per-slot tables."""
        new_reqs = scheduler_output.scheduled_new_reqs
        for new in new_reqs:
            sampling_params = new.sampling_params
            assert sampling_params is not None, "Pooling not supported in the v2 runner"

            req_id = new.req_id
            # A re-added id (streaming abort+resubmit) must clear its stale slot.
            self._remove_request(req_id)

            prompt_len = len(new.prompt_token_ids)
            # prefill_token_ids is the full known prefix (prompt + already-computed
            # tokens for resumed/PD requests); prompt_token_ids for a fresh req.
            all_token_ids = (
                new.prefill_token_ids
                if new.prefill_token_ids is not None
                else new.prompt_token_ids
            )
            self.req_states.add_request(
                req_id, prompt_len, all_token_ids, new.num_computed_tokens
            )
            slot = self.req_states.req_id_to_index[req_id]

            # Seeded requests carry their own generator; the rest use global RNG.
            generator = None
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(sampling_params.seed)
            self.sampling_states.add_request(slot, sampling_params, generator)

            self.block_table.add_row(new.block_ids, slot)

            if self.model_state is not None:
                # No-op for the common text path (rope_state is None).
                self.model_state.add_request(slot, new)

            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

        if new_reqs:
            # No-ops under the numpy substitution, kept for interface parity.
            self.req_states.apply_staged_writes()
            if self.model_state is not None:
                self.model_state.apply_staged_writes()

    def update_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Advance computed-token counts and append new blocks for running reqs."""
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            slot = self.req_states.req_id_to_index[req_id]
            num_computed = cached.num_computed_tokens[i]
            self.req_states.num_computed_tokens[slot] = num_computed
            # Clamp prefill progress to the prefill length (upstream update_requests).
            self.req_states.num_computed_prefill_tokens[slot] = min(
                num_computed, int(self.req_states.prefill_len[slot])
            )
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                self.block_table.append_row(new_block_ids, slot)

    def _order_scheduled_reqs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order this step's scheduled requests decodes-first (by token count).

        Returns parallel ``(slots, num_scheduled_tokens)`` arrays. Sorting by
        ascending scheduled-token count (upstream v2 convention) groups the
        1-token decodes ahead of the prefills so the multi-pass loop can emit
        pure decode passes (dispatched to the decode graph) before prefills.
        """
        slots: list[int] = []
        ntoks: list[int] = []
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            slot = self.req_states.req_id_to_index.get(req_id)
            if slot is None:
                continue
            slots.append(slot)
            ntoks.append(n)
        ntoks_np = np.array(ntoks, dtype=np.int32)
        order = np.argsort(ntoks_np, kind="stable")
        return np.array(slots, dtype=np.int32)[order], ntoks_np[order]

    def _select_batch(
        self,
        ordered_slots: np.ndarray,
        ordered_num_tokens: np.ndarray,
        start_index: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """Pick the sub-batch for one pass, applying the SMEM row caps.

        Mirrors the v1 fork's per-pass clamping over the decode-first ordering:
        a long request in the remaining tail forces the max-model-len row cap;
        prefill passes are additionally capped at ``max_prefill_num_reqs`` and the
        multi-pass loop picks up the rest. Returns ``(idx_mapping,
        num_scheduled_tokens, target_num_reqs, padded_query_len, end_index)``.
        """
        # A long request anywhere in the remaining tail forces max-model-len mode
        # (fewer rows fit SMEM), matching the v1 collection loop.
        use_max_model_len = self.most_model_len is None
        if not use_max_model_len and np.any(
            ordered_num_tokens[start_index:] > self.most_model_len
        ):
            use_max_model_len = True
        row_cap = (
            self.num_reqs_max_model_len
            if use_max_model_len
            else self.num_reqs_most_model_len
        )

        end_index = min(len(ordered_slots), start_index + row_cap)
        num_scheduled = ordered_num_tokens[start_index:end_index]

        # Prefill pass over the cap -> trim; the next pass takes the remainder.
        if (
            len(num_scheduled) > self.max_prefill_num_reqs
            and int(num_scheduled.max()) > 1
        ):
            end_index = start_index + self.max_prefill_num_reqs
            num_scheduled = ordered_num_tokens[start_index:end_index]

        idx_mapping = ordered_slots[start_index:end_index]
        max_scheduled = int(num_scheduled.max())

        if max_scheduled == 1:
            # Decode always runs at the max request bucket.
            target_num_reqs = self.max_num_reqs
        else:
            actual = len(num_scheduled)
            target_num_reqs = (
                self.min_num_reqs
                if actual <= self.min_num_reqs
                else self.max_prefill_num_reqs
            )

        padded_query_len = _get_padded_token_len(
            self.num_tokens_paddings, max_scheduled
        )
        return idx_mapping, num_scheduled, target_num_reqs, padded_query_len, end_index

    def _prepare_input_tokens(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        target_num_reqs: int,
        padded_query_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Host substitute for the v2 Triton input-prep kernels.

        Builds the per-step token tensors for a batch whose batch position ``b``
        maps to stable slot ``idx_mapping_np[b]`` and has
        ``num_scheduled_tokens[b]`` scheduled tokens. Replaces upstream's
        ``prepare_prefill_inputs`` + ``prepare_pos_seq_lens`` +
        ``combine_sampled_and_draft_tokens``; the last collapses to nothing on TT
        because decode tokens are read straight from ``all_token_ids`` (grown by
        the sampled-token writeback) rather than injected separately, and there is
        no spec decode.

        Returns 2D ``input_ids``/``positions`` ``[target_num_reqs,
        padded_query_len]`` (the TT forward layout; the runner copies these into
        persistent device buffers) plus the flat bookkeeping arrays
        ``query_start_loc`` ``[target_num_reqs + 1]``, ``seq_lens`` and
        ``logits_indices`` ``[target_num_reqs]``. Padding rows/columns stay zero.
        """
        num_reqs = len(idx_mapping_np)
        rs = self.req_states

        input_ids = np.zeros((target_num_reqs, padded_query_len), dtype=np.int32)
        positions = np.zeros((target_num_reqs, padded_query_len), dtype=np.int32)
        seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
        logits_indices = np.zeros(target_num_reqs, dtype=np.int32)

        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            n = int(num_scheduled_tokens[b])
            computed = int(rs.num_computed_tokens[slot])
            # Same gather for prefill and decode: the scheduled tokens for this
            # step are all_token_ids[computed : computed + n].
            input_ids[b, :n] = rs.all_token_ids[slot, computed : computed + n]
            positions[b, :n] = np.arange(n, dtype=np.int32) + computed
            seq_lens[b] = computed + n
            # One logit per request, at its last scheduled token (within-row).
            logits_indices[b] = n - 1

        query_start_loc = np.zeros(target_num_reqs + 1, dtype=np.int32)
        np.cumsum(num_scheduled_tokens, out=query_start_loc[1 : num_reqs + 1])
        # Non-decreasing pad tail (matches TTInputBatch.make_dummy).
        query_start_loc[num_reqs + 1 :] = query_start_loc[num_reqs]

        return input_ids, positions, query_start_loc, seq_lens, logits_indices

    def postprocess(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """Write sampled tokens back into ``TTRequestState`` (post_update substitute).

        Host substitute for upstream's ``post_update`` Triton kernel, for TT's
        no-spec-decode path (one token per sampling request). A request produces a
        real token only once it has consumed all its prefill (``seq_len >=
        prefill_len``); tokens from still-prefilling (partial-chunk) requests are
        discarded. The kept token is appended at ``total_len`` -- which equals
        ``seq_len`` at that point -- so the next step's input-prep gather
        (``all_token_ids[num_computed:...]``) reads it. ``num_computed_tokens`` is
        NOT advanced here: the scheduler supplies it via ``update_requests``.

        Returns the per-batch-position sampled token ids with discarded
        (still-prefilling) rows emptied, for the ModelRunnerOutput.
        """
        num_reqs = len(idx_mapping_np)
        rs = self.req_states
        valid: list[list[int]] = [list(row) for row in sampled_token_ids[:num_reqs]]

        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            seq_len = int(rs.num_computed_tokens[slot]) + int(num_scheduled_tokens[b])
            if seq_len < int(rs.prefill_len[slot]):
                # Still prefilling: ignore the sampled token from this partial req.
                valid[b] = []
                continue
            token_id = int(valid[b][0])
            pos = int(rs.total_len[slot])
            rs.all_token_ids[slot, pos] = token_id
            rs.total_len[slot] = pos + 1
            rs.last_sampled_tokens[slot, 0] = token_id

        return valid

    def _prepare_attn_tensors(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        seq_lens: np.ndarray,
        target_num_reqs: int,
        num_blocks_per_req: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host substitute for the v2 block-table / slot-mapping kernels.

        Builds the paged-attention tensors ``TTModelState.prepare_attn`` packages
        into a ``TTMetadata``: the read-path ``page_table``, the write-path
        ``fill_page_table``, and ``cache_position``. Batch position ``b`` maps to
        stable slot ``idx_mapping_np[b]``; the block table is gathered in batch
        order. Returns numpy arrays ``[target_num_reqs, num_blocks_per_req]`` /
        ``[target_num_reqs]`` (the runner copies them into persistent device
        buffers). Padding rows are null (page_table 0, cache_position -1).
        """
        num_reqs = len(idx_mapping_np)
        block_table_cpu = self.block_table[0].get_cpu_tensor()
        if hasattr(block_table_cpu, "numpy"):
            block_table_cpu = block_table_cpu.numpy()

        page_table = np.zeros((target_num_reqs, num_blocks_per_req), dtype=np.int32)
        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            page_table[b, :] = block_table_cpu[slot, :num_blocks_per_req]

        # Decode/write position per user; -1 for padding rows.
        cache_position = np.full(target_num_reqs, -1, dtype=np.int32)
        cache_position[:num_reqs] = seq_lens[:num_reqs] - 1

        # Prefix caching: roll each row so paged_fill_cache writes the suffix
        # blocks instead of overwriting shared prefix blocks. Per-row because
        # prefix lengths differ.
        offsets = np.zeros(num_reqs, dtype=np.int64)
        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            offsets[b] = (
                int(self.req_states.num_computed_tokens[slot]) // self.block_size
            )
        if np.any(offsets > 0):
            fill_page_table = page_table.copy()
            for b in range(num_reqs):
                if offsets[b] > 0:
                    fill_page_table[b] = np.roll(page_table[b], -int(offsets[b]))
        else:
            fill_page_table = page_table

        # A zero-scheduled (already-prefilled, re-batched) row would clobber its
        # real KV with padding; redirect its fill to the null block. The read
        # path keeps the real blocks.
        zero_rows = np.nonzero(num_scheduled_tokens[:num_reqs] == 0)[0]
        if len(zero_rows) > 0:
            if fill_page_table is page_table:
                fill_page_table = page_table.copy()
            fill_page_table[zero_rows, :] = 0

        return page_table, fill_page_table, cache_position

    # ------------------------------------------------------------------ #
    # Two-phase step driver (mirrors the v1 fork + upstream v2 split).
    # ------------------------------------------------------------------ #

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ):
        """Phase 1: apply the scheduler delta, then hand off to sample_tokens.

        Mirrors the v1 fork / upstream v2 two-phase contract (the worker calls
        execute_model then sample_tokens). All host state updates happen here via
        the tested lifecycle; the forward + sampling are deferred to
        sample_tokens. Returns None on a normal step (output comes from
        sample_tokens) or an empty output when nothing is scheduled.
        """
        assert (
            self.scheduler_output is None
        ), "execute_model called before sample_tokens consumed the prior step"
        # TT compiles one graph per (bucket) shape; keep dynamic shapes off so a
        # new shape recompiles rather than silently falling back.
        torch._dynamo.config.dynamic_shapes = False

        self.finish_requests(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens == 0:
            # No tokens this step (e.g. only KV-connector activity).
            from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.supports_mm_inputs:
            # Needs TTModelState.get_mm_embeddings + the encoder run (deferred).
            raise NotImplementedError(
                "multimodal path pending get_mm_embeddings (MRv2 Phase 3)."
            )

        self.scheduler_output = scheduler_output
        return None

    def sample_tokens(self, grammar_output):
        """Phase 2: run the (possibly multi-pass) batch loop and sample.

        Composes the tested host stages -- decode-first ordering, SMEM
        sub-batching, input-token prep, attention slot-mapping, and the
        sampled-token writeback -- around the single hardware leaf
        ``_run_model_pass``. The SMEM row caps mean a step may span several passes
        (start_index advances); results are concatenated in processing order.
        """
        if self.scheduler_output is None:
            # PP non-final rank / nothing stashed: output is unused.
            return None
        scheduler_output = self.scheduler_output
        self.scheduler_output = None

        if grammar_output is not None:
            raise NotImplementedError(
                "structured-output decoding pending for the v2 runner."
            )
        if self.num_prompt_logprobs:
            raise NotImplementedError("prompt logprobs pending for the v2 runner.")

        ordered_slots, ordered_num_tokens = self._order_scheduled_reqs(scheduler_output)

        out_req_ids: list[str] = []
        out_sampled: list[list[int]] = []

        start_index = 0
        while start_index < len(ordered_slots):
            (
                idx_mapping,
                num_scheduled,
                target_num_reqs,
                padded_query_len,
                end_index,
            ) = self._select_batch(ordered_slots, ordered_num_tokens, start_index)

            input_ids, positions, _query_start_loc, seq_lens, logits_indices = (
                self._prepare_input_tokens(
                    idx_mapping, num_scheduled, target_num_reqs, padded_query_len
                )
            )
            page_table, fill_page_table, cache_position = self._prepare_attn_tensors(
                idx_mapping,
                num_scheduled,
                seq_lens,
                target_num_reqs,
                self.max_num_blocks_per_req,
            )

            sampled = self._run_model_pass(
                idx_mapping,
                num_scheduled,
                target_num_reqs,
                padded_query_len,
                input_ids,
                positions,
                logits_indices,
                page_table,
                fill_page_table,
                cache_position,
            )

            valid = self.postprocess(idx_mapping, num_scheduled, sampled)
            for b in range(len(idx_mapping)):
                slot = int(idx_mapping[b])
                out_req_ids.append(self.req_states.index_to_req_id[slot])
                out_sampled.append(valid[b])

            start_index = end_index

        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(
            req_ids=out_req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(out_req_ids)},
            sampled_token_ids=out_sampled,
        )

    def _run_model_pass(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        target_num_reqs: int,
        padded_query_len: int,
        input_ids: np.ndarray,
        positions: np.ndarray,
        logits_indices: np.ndarray,
        page_table: np.ndarray,
        fill_page_table: np.ndarray,
        cache_position: np.ndarray,
    ) -> list[list[int]]:
        """Run one compiled forward + sample pass for the selected sub-batch.

        Copies the host arrays to the device, builds the attention metadata
        (``TTModelState.prepare_attn``) and sampling metadata
        (``XLASupportedSamplingMetadata.from_v2_states``), runs the compiled
        forward + sample, and returns the batch-ordered sampled token ids.
        Common text path only: grammar / cpu_sampling / prompt-logprobs are
        gated off in the driver.
        """
        from types import SimpleNamespace

        from .metadata import XLASupportedSamplingMetadata

        dev = self.device
        input_ids_dev = torch.from_numpy(input_ids).to(dev)
        positions_dev = torch.from_numpy(positions).to(dev)
        page_table_dev = torch.from_numpy(page_table).to(dev)
        fill_page_table_dev = torch.from_numpy(fill_page_table).to(dev)
        cache_position_dev = torch.from_numpy(cache_position).to(dev)
        logits_indices_dev = torch.from_numpy(logits_indices).to(dev)
        batch_idx_dev = torch.arange(target_num_reqs, dtype=torch.int32, device=dev)

        attn_metadata = self.model_state.prepare_attn(
            self.attention_layer_names,
            page_table_dev,
            cache_position_dev,
            fill_page_table=fill_page_table_dev,
            batch_idx=batch_idx_dev,
            num_users=target_num_reqs,
            dp_size=self.dp_size,
        )
        # from_v2_states only reads num_reqs + idx_mapping_np off the view.
        batch_view = SimpleNamespace(
            num_reqs=len(idx_mapping_np), idx_mapping_np=idx_mapping_np
        )
        sampling_metadata = XLASupportedSamplingMetadata.from_v2_states(
            self.req_states,
            self.sampling_states,
            batch_view,
            target_num_reqs,
            self.sampling_device,
            vocab_size=self.vocab_size,
        )

        selected = self._forward_and_sample(
            input_ids_dev,
            positions_dev,
            logits_indices_dev,
            attn_metadata,
            sampling_metadata,
        )
        # [target_num_reqs, 1] -> per active-req list; drop padding rows.
        return selected[: len(idx_mapping_np)].cpu().tolist()

    def _forward_and_sample(
        self, input_ids, positions, logits_indices, attn_metadata, sampling_metadata
    ):
        """Publish the attn metadata into the forward context, then run the graph."""
        from vllm.forward_context import set_forward_context

        num_tokens = input_ids.shape[0] * input_ids.shape[1]
        with set_forward_context(
            attn_metadata, self.vllm_config, num_tokens=num_tokens
        ):
            return self._forward_and_sample_compiled(
                input_ids, positions, logits_indices, sampling_metadata
            )

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def _forward_and_sample_compiled(
        self, input_ids, positions, logits_indices, sampling_metadata
    ):
        """Compiled model forward -> last-token select -> logits -> sample."""
        model_input_ids, model_positions, model_embeds, restore_shape = (
            self._prepare_model_call_tensors(input_ids, positions, None)
        )
        hidden_states = self.model(
            input_ids=model_input_ids,
            positions=model_positions,
            inputs_embeds=model_embeds,
        )
        hidden_states = self._restore_model_hidden_states(hidden_states, restore_shape)
        selected_states = self._select_hidden_states(hidden_states, logits_indices)
        logits = self.model.compute_logits(selected_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def _prepare_model_call_tensors(self, input_ids, positions, inputs_embeds):
        """Optionally flatten the 2D [reqs, tokens] tensors to 1D for flat_model_io."""
        if not self.use_flat_model_io:
            return input_ids, positions, inputs_embeds, None
        restore_shape = None
        if input_ids is not None and input_ids.ndim > 1:
            restore_shape = input_ids.shape
            input_ids = input_ids.reshape(-1)
        if inputs_embeds is not None and inputs_embeds.ndim > 2:
            if restore_shape is None:
                restore_shape = torch.Size(inputs_embeds.shape[:-1])
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        if positions.ndim > 1:
            if restore_shape is None:
                restore_shape = positions.shape
            positions = positions.reshape(-1)
        return input_ids, positions, inputs_embeds, restore_shape

    def _restore_model_hidden_states(self, hidden_states, restore_shape):
        if restore_shape is None or hidden_states.ndim != 2:
            return hidden_states
        return hidden_states.reshape(*restore_shape, hidden_states.shape[-1])

    def _select_hidden_states(self, hidden_states, indices_do_sample):
        # Gather each request's last-token hidden state: hidden is [reqs, tokens, H].
        batch_indices = torch.arange(indices_do_sample.shape[0], dtype=torch.int32)
        return hidden_states[batch_indices, indices_do_sample, :]

    def _sample_from_logits(self, logits, sampling_metadata):
        # Greedy fast-path (argmax) avoids the fused sampling kernel; the sampler
        # handles temperature/top-k/p/penalties/seeds otherwise.
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.no_logit_bias
            and sampling_metadata.no_bad_words
            and sampling_metadata.no_allowed_token_ids
            and sampling_metadata.no_min_tokens
            and sampling_metadata.no_generators
        ):
            return torch.argmax(logits, dim=-1, keepdim=True)
        return self.sampler(logits, sampling_metadata).sampled_token_ids

    # ------------------------------------------------------------------ #
    # KV cache allocation.
    # ------------------------------------------------------------------ #

    def _allocate_kv_caches(self, kv_cache_config: "KVCacheConfig") -> dict:
        """Allocate the per-layer KV cache tensors on the TT device.

        The device-coupled core of ``initialize_kv_cache`` (salvaged from the v1
        fork), split out so it can be exercised on-device without the engine
        wrappers. Standard attention gets separate ``[k_cache, v_cache]`` tensors
        (avoids slice/concat in the compiled graph); MLA gets a single latent
        tensor. Only one KV-cache group (no hybrid) and one owner per tensor are
        supported. Returns ``layer_name -> tensor | [k, v]``.
        """
        from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

        from .attention_impls.attention import TTAttentionBackend
        from .attention_impls.attention_mla import TTMLAAttentionBackend

        groups = kv_cache_config.kv_cache_groups
        if len(groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )
        if len(groups) == 0:
            # Valid when only a subset of layers is compiled (layer override).
            return {}

        assert groups[0].kv_cache_spec.block_size == self.block_size, (
            f"KV cache block_size {groups[0].kv_cache_spec.block_size} must match "
            f"the runner block_size {self.block_size}"
        )

        kv_cache_sizes: dict[str, int] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            assert (
                len(tensor.shared_by) == 1
            ), "A KV cache tensor shared by multiple layers is not supported on TT."
            kv_cache_sizes[tensor.shared_by[0]] = tensor.size

        kv_caches: dict = {}
        for group in groups:
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % spec.page_size_bytes == 0
                num_blocks = tensor_size // spec.page_size_bytes
                if isinstance(spec, MLAAttentionSpec):
                    # Single concatenated latent KV tensor (num_kv_heads == 1).
                    shape = TTMLAAttentionBackend.get_kv_cache_shape(
                        num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size
                    )
                    kv_caches[layer_name] = torch.zeros(shape, dtype=spec.dtype).to(
                        self.device
                    )
                elif isinstance(spec, AttentionSpec):
                    if self.enable_tensor_parallel:
                        tp_size = self.original_parallel_config.tensor_parallel_size
                        assert spec.num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {spec.num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD"
                        )
                    shape = TTAttentionBackend.get_kv_cache_shape(
                        num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size
                    )
                    # spec.dtype may be a 1-byte accounting dtype; the device
                    # buffers use the real transfer dtype.
                    k = torch.zeros(shape, dtype=self.kv_cache_dtype).to(self.device)
                    v = torch.zeros(shape, dtype=self.kv_cache_dtype).to(self.device)
                    kv_caches[layer_name] = [k, v]
                else:
                    raise NotImplementedError(
                        f"Unsupported KV cache spec: {type(spec)}"
                    )
        return kv_caches

    def get_kv_cache_spec(self) -> dict:
        """Build the per-layer KVCacheSpec from the model's attention modules.

        Salvaged from the v1 fork: walks the attention layers in the static
        forward context and emits a Full / SlidingWindow / MLA spec each,
        skipping cross-layer-shared (recorded in ``shared_kv_cache_layers``) and
        encoder-only layers. The engine uses this to budget KV blocks
        (``determine_available_memory``) and to build the ``KVCacheConfig`` passed
        back to ``initialize_kv_cache``.
        """
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
            SlidingWindowSpec,
        )

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        kv_cache_spec: dict = {}
        for layer_name, attn_module in layers.items():
            if isinstance(attn_module, Attention):
                if attn_module.kv_sharing_target_layer_name is not None:
                    # Reuses another layer's KV cache; allocate nothing here.
                    self.shared_kv_cache_layers[layer_name] = (
                        attn_module.kv_sharing_target_layer_name
                    )
                    continue
                if attn_module.attn_type == AttentionType.DECODER:
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_spec_dtype,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_spec_dtype,
                        )
                elif attn_module.attn_type in (
                    AttentionType.ENCODER,
                    AttentionType.ENCODER_ONLY,
                ):
                    continue  # encoder-only attention needs no KV cache
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError(
                        "Encoder-decoder attention is not supported yet."
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_spec_dtype,
                    cache_dtype_str=cache_dtype_str,
                )
        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig") -> None:
        """Allocate KV caches and bind them into the forward context.

        Wraps ``_allocate_kv_caches`` with the engine integration. The common
        single-device path allocates and binds; the SPMD KV-cache sharding and
        KV-transfer registration (salvage v1 initialize_kv_cache 3436-3459) land
        with the multi-device ``__init__``. ``self.kv_caches`` (a list) and
        ``self.vllm_config`` are provided by ``__init__``.
        """
        from vllm.v1.worker.utils import bind_kv_cache

        kv_caches = self._allocate_kv_caches(kv_cache_config)
        if not kv_caches:
            return
        self.kv_cache_config = kv_cache_config
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

    # ------------------------------------------------------------------ #
    # Worker-facing shims + warmup.
    # ------------------------------------------------------------------ #

    def get_model(self):
        return self.model

    def reset_mm_cache(self) -> None:
        if self.mm_budget is not None:
            self.mm_budget.reset_cache()

    def add_lora(self, lora_request) -> bool:
        raise NotImplementedError("LoRA is not supported in the v2 runner yet.")

    @contextmanager
    def maybe_setup_dummy_loras(self, lora_config):
        """No-op LoRA warmup context (LoRA deferred). Raises if LoRA is on."""
        if lora_config is not None:
            raise NotImplementedError("LoRA is not supported in the v2 runner yet.")
        yield

    def get_supported_tasks(self) -> tuple:
        """Report the generation tasks this model supports (via TTModelState)."""
        if self.model_config.runner_type != "generate":
            raise NotImplementedError(
                "Only the generate runner type is supported in the v2 runner."
            )
        assert (
            self.model_state is not None
        ), "load_model must run before get_supported_tasks"
        return tuple(self.model_state.get_supported_generation_tasks())

    def reset_dynamo_cache(self) -> None:
        """Clear the compiled-model dynamo cache so it re-traces cleanly."""
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        if self.model_config.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWithNoGuardsWrapper):
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object()
            )
            compiled_model.compiled = False
            TorchCompileWithNoGuardsWrapper.__init__(compiled_model)

    def update_config(self, overrides: dict) -> None:
        from vllm.config import update_config as _update_config

        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            setattr(
                self,
                config_name,
                _update_config(getattr(self, config_name), config_overrides),
            )

    def reload_weights(self) -> None:
        raise NotImplementedError(
            "reload_weights is not supported in the v2 runner yet."
        )

    def ensure_kv_transfer_shutdown(self) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        from vllm.distributed.kv_transfer.kv_transfer_state import (
            ensure_kv_transfer_shutdown,
        )

        if has_kv_transfer_group():
            ensure_kv_transfer_shutdown()

    def _warmup_buckets(self) -> list[tuple[int, int]]:
        """(target_num_reqs, padded_query_len) pairs to precompile.

        Decode always runs at max_num_reqs; prefill runs at the min / max prefill
        buckets. Each request-count bucket is compiled at every token-length
        padding, matching the shapes _select_batch can produce at runtime.
        """
        targets = sorted(
            {self.max_num_reqs, self.min_num_reqs, self.max_prefill_num_reqs}
        )
        return [(t, q) for t in targets for q in self.num_tokens_paddings]

    def capture_model(self) -> None:
        """Precompile the forward/sample graph at every runtime bucket shape.

        Avoids paying compile latency on the first real request. Warms the greedy
        path (argmax) at each (num_reqs, query_len) bucket; the random-sampling,
        multimodal, and cpu_sampling warmups are deferred with those runtime
        paths. Needs the model + KV cache initialised, so it runs at stand-up.
        """
        if self.cpu_sampling:
            raise NotImplementedError("cpu_sampling warmup path is not ported yet.")
        torch._dynamo.config.dynamic_shapes = False
        for target_num_reqs, padded_query_len in self._warmup_buckets():
            self._precompile_bucket(target_num_reqs, padded_query_len)

    def _precompile_bucket(self, target_num_reqs: int, padded_query_len: int) -> None:
        from .metadata import XLASupportedSamplingMetadata

        dev = self.device
        input_ids = torch.zeros(
            (target_num_reqs, padded_query_len), dtype=torch.int32, device=dev
        )
        positions = torch.zeros(
            (target_num_reqs, padded_query_len), dtype=torch.int32, device=dev
        )
        logits_indices = torch.zeros(target_num_reqs, dtype=torch.int32, device=dev)
        page_table = torch.zeros(
            (target_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=dev,
        )
        cache_position = torch.zeros(target_num_reqs, dtype=torch.int32, device=dev)
        batch_idx = torch.arange(target_num_reqs, dtype=torch.int32, device=dev)

        attn_metadata = self.model_state.prepare_attn(
            self.attention_layer_names,
            page_table,
            cache_position,
            fill_page_table=page_table,
            batch_idx=batch_idx,
            num_users=target_num_reqs,
            dp_size=self.dp_size,
        )
        # Defaults are all-greedy -> warms the argmax fast-path.
        sampling_metadata = XLASupportedSamplingMetadata(all_greedy=True)
        self._forward_and_sample(
            input_ids, positions, logits_indices, attn_metadata, sampling_metadata
        )

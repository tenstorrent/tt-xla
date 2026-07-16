# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT Model Runner v2 (MRv2) -- fork of upstream ``vllm/v1/worker/gpu/model_runner.py``.

Phase 3 of the MRv2 adoption. Upstream's v2 runner is Triton/CUDA/UVA-native
throughout, so TT forks it and substitutes host-side mechanisms rather than
subclassing ``GPUModelRunner``.

This file is built in reviewable sub-increments. Present so far: the request
lifecycle, the decode-first batch selection / SMEM multi-pass clamping, the host
input-token preparation, and the attention slot-mapping build (feeding
``TTModelState.prepare_attn``). Still to land: ``__init__``/``load_model``
(construct the state below), ``initialize_kv_cache``, the compiled forward
graphs, and ``execute_model``/``sample_tokens``.

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
sub-increment): ``req_states``, ``sampling_states``, ``block_table`` (a vLLM
``MultiGroupBlockTable``), ``encoder_cache`` (dict), ``num_prompt_logprobs``
(dict), ``model_state`` (or None), and ``vocab_size``.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingType

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """First padding >= x (the per-step query-length bucket)."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


class TTModelRunnerV2:
    """Tenstorrent MRv2 model runner (lifecycle sub-increment; see module doc)."""

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

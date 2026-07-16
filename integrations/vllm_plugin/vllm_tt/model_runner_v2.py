# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT Model Runner v2 (MRv2) -- fork of upstream ``vllm/v1/worker/gpu/model_runner.py``.

Phase 3 of the MRv2 adoption. Upstream's v2 runner is Triton/CUDA/UVA-native
throughout, so TT forks it and substitutes host-side mechanisms rather than
subclassing ``GPUModelRunner``.

This file is built in reviewable sub-increments; only the request lifecycle is
present so far. Still to land: ``__init__``/``load_model`` (construct the state
below), ``initialize_kv_cache``, host ``prepare_inputs`` (the Triton-kernel
substitutes filling ``TTInputBatch``), the compiled forward graphs, and
``execute_model``/``sample_tokens``.

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

from typing import TYPE_CHECKING

import torch

from vllm.sampling_params import SamplingType

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


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

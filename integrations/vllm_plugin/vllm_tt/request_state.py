# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT ``RequestState`` for the vLLM v2 model runner.

Each request owns a fixed slot for its lifetime; the slot returns to a free list
on removal (no condensing). Holds only token bookkeeping -- sampling params live
in ``TTSamplingStates`` and block tables in the runner. Storage is host-side
numpy, so ``apply_staged_writes`` is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class TTRequestState:
    """Structure-of-arrays keyed by ``req_state_idx``, a stable per-request slot."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: "torch.device",
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        self.all_token_ids = np.zeros(
            (self.max_num_reqs, self.max_model_len), dtype=np.int32
        )

        # prompt_len: tokens in the user prompt.
        # prefill_len: tokens fed to the runner (>= prompt_len; larger on resume
        #   after preemption). Prompt vs output tokens must be distinguished for
        #   prompt logprobs and frequency penalties.
        self.prompt_len = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.prefill_len = np.zeros(self.max_num_reqs, dtype=np.int32)
        # total_len = prompt_len + output_len; grows as the request progresses.
        self.total_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled token per slot (seeds the next decode input).
        self.last_sampled_tokens = np.zeros((self.max_num_reqs, 1), dtype=np.int64)

        # Draft tokens (spec decode); zero-width when num_speculative_steps == 0.
        self.draft_tokens = np.zeros(
            (self.max_num_reqs, self.num_speculative_steps), dtype=np.int64
        )
        # How many of draft_tokens[slot] are live this step (drafts vary in length).
        self.num_draft_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)

        self.next_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.prompt_len[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert (
            prefill_len >= prompt_len
        ), f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        self.prefill_len[req_idx] = prefill_len
        self.total_len[req_idx] = prefill_len
        self.all_token_ids[req_idx, :prefill_len] = all_token_ids
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens[req_idx] = num_computed_tokens

        if 0 < num_computed_tokens <= prefill_len:
            # PD disagg / resumed requests: seed last_sampled with the last
            # computed token so the first decode step reads the right input id.
            # Fresh prefill (num_computed_tokens == 0) never reads this slot.
            self.last_sampled_tokens[req_idx] = all_token_ids[num_computed_tokens - 1]
        self.draft_tokens[req_idx] = 0
        self.num_draft_tokens[req_idx] = 0

    def set_draft_tokens(self, req_idx: int, drafts: list[int]) -> None:
        """Stage this step's drafts for a slot (spec decode).

        Drafts land in all_token_ids right after the committed tokens so the
        input gather picks them up, but total_len does NOT advance: they are
        unverified. Accepted tokens overwrite them in the writeback.
        """
        n = len(drafts)
        assert n <= self.num_speculative_steps, (
            f"{n} drafts exceeds num_speculative_steps " f"{self.num_speculative_steps}"
        )
        self.num_draft_tokens[req_idx] = n
        if n:
            self.draft_tokens[req_idx, :n] = drafts
            pos = int(self.total_len[req_idx])
            self.all_token_ids[req_idx, pos : pos + n] = drafts

    def clear_draft_tokens(self, req_idx: int) -> None:
        self.num_draft_tokens[req_idx] = 0

    def apply_staged_writes(self) -> None:
        """No-op: numpy writes land immediately."""
        return None

    def remove_request(self, req_id: str) -> bool:
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return False
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return True

    def is_prefilling(self, idx_mapping_np: np.ndarray) -> np.ndarray:
        return (
            self.num_computed_prefill_tokens[idx_mapping_np]
            < self.prefill_len[idx_mapping_np]
        )

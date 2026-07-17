# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT ``InputBatch`` for vLLM Model Runner v2 (MRv2).

Phase 2 of the MRv2 adoption: the transient per-step batch view. Mirrors
upstream ``vllm.v1.worker.gpu.input_batch`` (as of vLLM v0.22.1) -- the
``InputBuffers`` (persistent device scratch) and the ``InputBatch`` dataclass
(a fresh per-step view the runner populates).

Scope / status
--------------
* ``InputBatch`` is a pure view: the runner rebuilds it every step from
  ``TTRequestState`` (see request_state.py). Fields tied to features TT does
  not support -- ``expanded_idx_mapping`` / ``expanded_local_pos`` /
  ``num_draft_tokens*`` (spec decode) and ``dcp_local_seq_lens`` (DCP) -- are
  kept for structural parity but degenerate to the no-spec / no-DCP path.
* Upstream's module also holds the Triton input-prep kernels
  (``prepare_prefill_inputs``, ``prepare_pos_seq_lens``,
  ``combine_sampled_and_draft_tokens``, ``expand_idx_mapping``, ``post_update``,
  ...). Those are input-prep, called from the runner's ``prepare_inputs``, and
  have no Triton equivalent on TT. They are deferred to Phase 3, where the TT
  runner fork substitutes host-side numpy / torch (salvage from the v1 runner's
  ``_prepare_inputs``). We do NOT ship stubs for them here.
* Like request_state.py, this module has no ``vllm.v1.worker.gpu.*`` dependency
  and is import-safe, but is intentionally NOT imported from the package
  ``__init__`` until the TT v2 runner (Phase 3) wires it in. UNVALIDATED at
  runtime until then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from vllm.utils import random_uuid


class TTInputBuffers:
    """Persistent device-side scratch buffers, owned by the runner.

    Sliced into per-step ``TTInputBatch`` views. Mirrors upstream
    ``InputBuffers``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)


@dataclass
class TTInputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding (unused on TT).
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit.
    expanded_local_pos: torch.Tensor

    # [num_reqs] batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests (0 on TT: no spec decode).
    num_draft_tokens: int
    # [num_reqs] draft tokens scheduled per request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens.
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs] DCP per-request local seq_lens (None on TT: no DCP).
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs] CPU bool array.
    is_prefilling_np: np.ndarray

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any request in the batch uses structured output.
    has_structured_output_reqs: bool

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: TTInputBuffers,
    ) -> "TTInputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals query_len for the dummy (fresh-prefill shape).
        input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens[num_reqs - 1] += num_tokens % num_reqs
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
        )

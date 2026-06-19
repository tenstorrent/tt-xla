# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness test for the #4278 incremental output-token-count fix.

Validates that InputBatch.update_output_token_counts() (incremental, persistent)
produces exactly the same [num_reqs, vocab] tensor as the original from-scratch
XLASupportedSamplingMetadata._compute_token_counts(), across the full slot
lifecycle: add_request (incl. resumed requests with prior output), token
appends, swap_states, remove_request + condense.

Run from the tt-xla venv:
  cd /home/kmabee/tt-xla && source venv/activate
  cd integrations/vllm_plugin && python test_incremental_counts.py
"""

import types

import torch
from vllm_tt.input_batch import InputBatch
from vllm_tt.metadata import XLASupportedSamplingMetadata

VOCAB = 32
MAX_REQS = 4
PADDED = MAX_REQS


def make_request(req_id, prompt_ids, output_ids, rep_penalty=1.1):
    sp = types.SimpleNamespace(
        sampling_type=2,  # not GREEDY -> exercises random path; value unused here
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=rep_penalty,
        min_tokens=0,
        all_stop_token_ids=set(),
        logprobs=None,
        prompt_logprobs=None,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
    )
    return types.SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(prompt_ids),
        prompt_embeds=None,
        output_token_ids=list(output_ids),  # live list; we append to it
        num_tokens=len(prompt_ids) + len(output_ids),
        num_computed_tokens=len(prompt_ids),
        block_ids=[],
        sampling_params=sp,
        pooling_params=None,
        generator=None,
        lora_request=None,
    )


def reference(ib):
    """From-scratch counts using the original static method."""
    return XLASupportedSamplingMetadata._compute_token_counts(
        ib.req_output_token_ids, PADDED, VOCAB
    )


def check(ib, label):
    got = ib.update_output_token_counts(PADDED).clone()
    ref = reference(ib)
    # Only the active rows [0:num_reqs] must match; padding rows are penalty
    # no-ops (default coeffs), but the active rows are what matters.
    nr = ib.num_reqs
    ok = torch.equal(got[:nr], ref[:nr])
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  num_reqs={nr}")
    if not ok:
        for i in range(nr):
            if not torch.equal(got[i], ref[i]):
                gnz = {int(v): int(got[i, v]) for v in got[i].nonzero().flatten()}
                rnz = {int(v): int(ref[i, v]) for v in ref[i].nonzero().flatten()}
                print(f"    row {i}: incremental={gnz} reference={rnz}")
        raise SystemExit(1)
    return got


def new_ib():
    return InputBatch(
        max_num_reqs=MAX_REQS,
        max_model_len=64,
        max_num_batched_tokens=256,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=VOCAB,
        block_sizes=[],
        kernel_block_sizes=[],
        is_pooling_model=False,
    )


def main():
    torch.manual_seed(0)
    ib = new_ib()

    # 1. Add 3 requests; one (r2) resumes with pre-existing output history.
    r0 = make_request("r0", [1, 2, 3], [])
    r1 = make_request("r1", [4, 5], [])
    r2 = make_request("r2", [6], [7, 7, 8])  # resumed: 7 twice, 8 once
    ib.add_request(r0)
    ib.add_request(r1)
    ib.add_request(r2)
    check(ib, "after add (r2 resumed with [7,7,8])")

    # 2. Append decode tokens to the live output lists (as the runner does).
    for tok in [9, 9, 9]:
        r0.output_token_ids.append(tok)
    for tok in [10, 11]:
        r1.output_token_ids.append(tok)
    r2.output_token_ids.append(7)  # 7 now appears 3x for r2
    check(ib, "after appends")

    # 3. More appends, then check again (incremental cursor advances).
    for tok in [9, 12]:
        r0.output_token_ids.append(tok)
    check(ib, "after more appends")

    # 4. Swap slots 0 and 2.
    ib.swap_states(0, 2)
    check(ib, "after swap_states(0,2)")
    # append after swap to make sure cursors followed the rows
    r0.output_token_ids.append(13)  # r0 now lives in slot 2
    r2.output_token_ids.append(8)  # r2 now lives in slot 0
    check(ib, "after post-swap appends")

    # 5. Remove the middle request (r1, slot 1) then condense.
    idx = ib.remove_request("r1")
    ib.condense([idx])
    check(ib, "after remove(r1)+condense")
    # append after condense
    for tok in [9, 9]:
        r0.output_token_ids.append(tok)
    r2.output_token_ids.append(7)
    check(ib, "after post-condense appends")

    # 6. Add a fresh request into the freed slot; ensure its row was reset
    #    (not polluted by the previously-evicted r1's counts).
    r3 = make_request("r3", [20], [21])
    ib.add_request(r3)
    check(ib, "after add r3 into reused slot")
    r3.output_token_ids.append(21)
    r3.output_token_ids.append(22)
    check(ib, "after r3 appends")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

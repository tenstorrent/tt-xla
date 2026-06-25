# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal single-device sanity reproducer for the Llasa-8B Out-Of-Memory failure.

Tracked in https://github.com/tenstorrent/tt-xla/issues/5371.

The full model test
    test_all_models_torch[llasa/causal_lm/pytorch-Llasa-8B-single_device-inference]
compiles successfully but fails at runtime with a TT_FATAL while moving a weight
onto device DRAM:

    Out of Memory: Not enough space to allocate 1587609600 B DRAM buffer across 12
    banks, where each bank needs to store 132300800 B, but bank size is 1070773184 B

This is a single-chip capacity limit, not a compiler bug. Llasa-8B has
vocab_size=193800, hidden_size=4096 and tie_word_embeddings=False, so both
``model.embed_tokens.weight`` and ``lm_head.weight`` are [193800, 4096] bf16 =
193800 * 4096 * 2 = 1,587,609,600 B each -- exactly the failing allocation. A
single wormhole_b0 has 12 DRAM banks * 1,070,773,184 B ~= 12 GB, far short of the
model's ~17 GB bf16 footprint.

This test rebuilds just the embedding-class weights (the actual offending tensors)
plus a filler parameter standing in for the 32 transformer layers, so the total
device footprint exceeds DRAM and the same bank_manager.cpp:462 OOM is raised --
without the 32 transformer layers or the HuggingFace checkpoint download.
"""

import pytest
import torch
import torch_xla.core.xla_model as xm

# Llasa-8B (HKUSTAudio/Llasa-8B) config values driving the OOM.
LLASA_VOCAB = 193800
LLASA_HIDDEN = 4096
LLASA_SEQ_LEN = 182  # sequence length seen in the dumped IR (tensor<1x182x193800xbf16>)

# embed_tokens / lm_head weight size: 193800 * 4096 * 2 bytes = 1,587,609,600 B.
EMBEDDING_WEIGHT_BYTES = LLASA_VOCAB * LLASA_HIDDEN * 2
assert EMBEDDING_WEIGHT_BYTES == 1_587_609_600

# Filler rows standing in for the ~10 GB of transformer-layer weights. Combined
# with the two ~1.59 GB embedding-class weights this pushes the resident footprint
# (~13.8 GB) past a single wormhole_b0's ~12 GB DRAM, guaranteeing the OOM.
FILLER_ROWS = 1_300_000


@pytest.mark.single_device
def test_llasa_8b_embedding_oom():
    """Reproduces the Llasa-8B single-chip DRAM OOM (issue #5371)."""
    device = xm.xla_device()

    # The exact Llasa-8B embedding-class weights (untied): 1.587 GB each.
    embed = torch.nn.Embedding(LLASA_VOCAB, LLASA_HIDDEN, dtype=torch.bfloat16)
    lm_head = torch.nn.Linear(
        LLASA_HIDDEN, LLASA_VOCAB, bias=False, dtype=torch.bfloat16
    )
    # Stand-in for the transformer-layer weights so the total exceeds device DRAM.
    filler = torch.nn.Parameter(
        torch.zeros(FILLER_ROWS, LLASA_HIDDEN, dtype=torch.bfloat16)
    )

    input_ids = torch.zeros(1, LLASA_SEQ_LEN, dtype=torch.long)

    with pytest.raises(RuntimeError, match="Out of Memory"):
        embed = embed.to(device)
        lm_head = lm_head.to(device)
        dev_filler = filler.to(device)
        ids = input_ids.to(device)

        hidden = embed(ids)  # [1, 182, 4096]
        # Touch the full filler so it must stay resident in DRAM alongside the
        # embedding weights (mirrors weights all being live program inputs).
        hidden = hidden + dev_filler.sum()
        logits = lm_head(hidden)  # [1, 182, 193800]

        # Force graph execution -> input tensors moved to device -> OOM.
        _ = logits.to("cpu")
        xm.mark_step()

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test argmax with different vocab sizes under SPMD."""

import os

os.environ["XLA_STABLEHLO_COMPILE"] = "1"

import torch
import torch_xla
import torch_xla.runtime as xr

xr.use_spmd()
device = torch_xla.device()

for vocab_size in [32000, 151936]:

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def argmax_fn(logits):
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = torch.randn(1, vocab_size, dtype=torch.bfloat16).to(device)
    logits[0, vocab_size // 2] = 100.0
    expected = logits.cpu().argmax(dim=-1, keepdim=True).item()
    result = argmax_fn(logits)
    print(
        f"vocab_size={vocab_size}: expected={expected}, got={result.cpu().item()}, match={expected == result.cpu().item()}"
    )

    # Reset for next iteration
    torch._dynamo.reset()

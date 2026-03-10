# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro for argmax returning 0 under SPMD torch.compile."""

import os

os.environ["XLA_STABLEHLO_COMPILE"] = "1"

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Enable SPMD
xr.use_spmd()

device = xm.xla_device()
print(f"Device: {device}")
print(f"SPMD enabled: {xr.is_spmd()}")

# Create a logits tensor similar to what vLLM produces
# Shape: [1, 32000] (batch=1, vocab_size=32000)
logits_cpu = torch.randn(1, 32000, dtype=torch.bfloat16)
# Make position 4123 the clear maximum
logits_cpu[0, 4123] = 100.0

expected = logits_cpu.argmax(dim=-1, keepdim=True)
print(f"Expected argmax: {expected}")

logits = logits_cpu.to(device)

# Test 1: Eager argmax on XLA device
eager_result = torch.argmax(logits, dim=-1, keepdim=True)
xm.mark_step()
print(f"Eager XLA argmax: {eager_result.cpu()}")


# Test 2: Compiled argmax (no SPMD mesh)
@torch.compile(backend="tt", fullgraph=True, dynamic=False)
def compiled_argmax_simple(x):
    return torch.argmax(x, dim=-1, keepdim=True)


result_compiled = compiled_argmax_simple(logits)
print(f"Compiled argmax (no mesh): {result_compiled.cpu()}")

import numpy as np

# Test 3: With SPMD mesh and sharding
from torch_xla.distributed.spmd import Mesh

mesh_shape = (1, 2)
device_ids = np.array(range(2))
mesh = Mesh(device_ids, mesh_shape, ("dp", "mp"))

# Apply sharding then replicate (mimicking what model_runner does)
from torch_xla.distributed.spmd import mark_sharding

# Replicate the logits (None, None means replicate both dims)
mark_sharding(logits, mesh, (None, None))

result_sharded = compiled_argmax_simple(logits)
print(f"Compiled argmax (with SPMD mesh, replicated): {result_sharded.cpu()}")


# Test 4: What about with the XLASupportedSamplingMetadata pattern?
# Mimic the full sample_from_logits structure
class FakeMeta:
    all_greedy = True
    no_penalties = True
    no_logit_bias = True
    no_bad_words = True
    no_allowed_token_ids = True
    no_min_tokens = True
    no_generators = True


@torch.compile(backend="tt", fullgraph=True, dynamic=False)
def sample_like_vllm(logits, meta):
    if (
        meta.all_greedy
        and meta.no_penalties
        and meta.no_logit_bias
        and meta.no_bad_words
        and meta.no_allowed_token_ids
        and meta.no_min_tokens
        and meta.no_generators
    ):
        return torch.argmax(logits, dim=-1, keepdim=True)
    return logits  # won't reach


meta = FakeMeta()
result_vllm = sample_like_vllm(logits, meta)
print(f"Compiled sample_like_vllm (SPMD): {result_vllm.cpu()}")

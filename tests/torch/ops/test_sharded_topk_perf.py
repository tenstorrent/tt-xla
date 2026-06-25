# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated A/B microbench for issue #4494: "move topk chunking for sampling into
tt-mlir".

It compiles ONLY the sampler candidate-building subgraph (no model weights) on a
vocab-sharded (None, "model") mesh, in two variants on the SAME build:

  legacy  : sharding_constraint -> (None, None)         [full-vocab all_gather]
            then chunked multi-core topk over the replicated vocab
            (reproduces main's pre-sharded-topk path).

  sharded : composite_topk on the (None, "model") logits  [local topk per shard
            -> all_gather of the tiny [B, 128] candidate set -> merge]
            (the new tt-mlir custom-sharding-rule path).

Verify the issue's device-time claim by running each variant under tracy and
diffing the summed DEVICE FW DURATION of the topk / CCL / concat ops:

  tracy -p -r --sync-host-device -m pytest -svv \
      tests/torch/ops/test_sharded_topk_perf.py -k legacy
  tracy -p -r --sync-host-device -m pytest -svv \
      tests/torch/ops/test_sharded_topk_perf.py -k sharded

On lb (8 chips) a (1, 8) mesh gives 16032-wide shards == galaxy's per-shard
width, so per-shard topk device time is directly comparable to galaxy [4, 8].

Env knobs:
  TOPK_VOCAB   vocab size            (default 128256, Llama-3.1)
  TOPK_BATCH   batch rows            (default 32, the ttnn.sampling kernel size)
"""

import math
import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tt_torch.composite_ops import composite_topk
from tt_torch.sharding import sharding_constraint_tensor

# --- constants mirrored from integrations/vllm_plugin/vllm_tt/sampler.py ---
_TOPK_MAX_CHUNK_SIZE = 32768  # largest power-of-2 below 65536
_TOPK_K_PER_CHUNK = 32  # candidates kept per chunk
_TTNN_SAMPLING_BATCH_SIZE = 32  # ttnn.sampling kernel requires batch=32
_SHARDED_TOPK_CANDIDATES = 128  # matches typical num_chunks * k_per_chunk


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _get_topk_split_params(vocab_size: int) -> tuple[int, int]:
    num_chunks = math.ceil(vocab_size / _TOPK_MAX_CHUNK_SIZE)
    chunk_size = math.ceil(vocab_size / num_chunks)
    return chunk_size, _next_power_of_2(chunk_size)


def _chunked_topk_candidates(logits, *, vocab_sharded):
    """Verbatim copy of sampler.chunked_topk_candidates (both paths)."""
    batch = logits.shape[0]
    logits = torch.nn.functional.pad(
        logits,
        (0, 0, 0, _TTNN_SAMPLING_BATCH_SIZE - batch),
        value=float("-inf"),
    )

    if vocab_sharded:
        k = int(os.environ.get("TOPK_K", str(_SHARDED_TOPK_CANDIDATES)))
        all_values, all_indices = composite_topk(
            logits, k=k, dim=-1, largest=True, sorted=False
        )
        return all_values[:batch], all_indices[:batch]

    chunk_size, padded_chunk_size = _get_topk_split_params(logits.shape[-1])
    chunks = torch.split(logits, chunk_size, dim=-1)
    topk_values_list = []
    topk_indices_list = []
    for i, chunk in enumerate(chunks):
        if chunk.shape[-1] < padded_chunk_size:
            chunk = torch.nn.functional.pad(
                chunk, (0, padded_chunk_size - chunk.shape[-1]), value=float("-inf")
            )
        vals, inds = torch.topk(chunk, k=_TOPK_K_PER_CHUNK, dim=-1)
        topk_values_list.append(vals)
        topk_indices_list.append(inds + i * chunk_size)

    all_values = torch.cat(topk_values_list, dim=-1)
    all_indices = torch.cat(topk_indices_list, dim=-1)

    cur_w = all_values.shape[-1]
    target_w = max(64, _next_power_of_2(cur_w))
    if cur_w < target_w:
        pad = target_w - cur_w
        all_values = torch.nn.functional.pad(all_values, (0, pad), value=float("-inf"))
        all_indices = torch.nn.functional.pad(all_indices, (0, pad), value=0)

    return all_values[:batch], all_indices[:batch]


class _CandidateBuilder(torch.nn.Module):
    def __init__(self, mesh, *, vocab_sharded, replicate_first):
        super().__init__()
        self.mesh = mesh
        self.vocab_sharded = vocab_sharded
        self.replicate_first = replicate_first

    def forward(self, logits):
        if self.replicate_first:
            # main's compute_logits: gather the (None,"model") vocab to replicated.
            logits = sharding_constraint_tensor(logits, self.mesh, (None, None))
        return _chunked_topk_candidates(logits, vocab_sharded=self.vocab_sharded)


def _candidate_comparator(device_output, golden_output, args, kwargs):
    """top-k ordering is non-deterministic; compare gathered values, not order."""
    device_vals, device_idx = device_output
    golden_vals, golden_idx = golden_output
    batch = device_idx.shape[0]
    input_tensor = args[0][:batch].cpu()
    dgath = torch.gather(input_tensor, -1, device_idx.cpu().clamp(min=0))
    ggath = torch.gather(input_tensor, -1, golden_idx.cpu().clamp(min=0))
    cos = torch.nn.functional.cosine_similarity(
        dgath.flatten().unsqueeze(0).float(), ggath.flatten().unsqueeze(0).float()
    )
    print(f"\n  candidate values_cos_sim={cos.item():.6f}")
    assert cos > 0.99, f"candidate values wrong: cos_sim={cos.item():.6f}"


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    "path",
    [
        pytest.param("legacy", id="legacy_replicate_chunked"),
        pytest.param("sharded", id="sharded_composite"),
    ],
)
def test_sharded_topk_candidates_perf(path):
    vocab = int(os.environ.get("TOPK_VOCAB", "128256"))
    batch = int(os.environ.get("TOPK_BATCH", "32"))
    num_devices = xr.global_runtime_device_count()
    # MESH_SHAPE="4,8" reproduces galaxy's 2D [4,8] vocab sharding (model axis=8,
    # 16032-wide shards). Default (1, num_devices): on lb (8 chips) this is (1,8),
    # which already gives 8-way / 16032-wide shards == galaxy [4,8] per-shard width.
    env_mesh = os.environ.get("MESH_SHAPE")
    mesh_shape = tuple(int(x) for x in env_mesh.split(",")) if env_mesh else (1, num_devices)
    model_axis = mesh_shape[1]
    assert vocab % model_axis == 0, f"vocab {vocab} not divisible by model axis {model_axis}"
    mesh = Mesh(list(range(num_devices)), mesh_shape, ("batch", "model"))

    module = _CandidateBuilder(
        mesh,
        vocab_sharded=(path == "sharded"),
        replicate_first=(path == "legacy"),
    )

    def shard_spec_fn(model, args, kwargs):
        return {args[0]: (None, "model")}

    run_graph_test(
        module,
        [torch.randn(batch, vocab, dtype=torch.float32)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        custom_comparator=_candidate_comparator,
    )

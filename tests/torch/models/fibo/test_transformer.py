# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FIBO — BriaFiboTransformer2DModel (DiT) component test (8B)."""

import os
import time

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.fibo.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy
def test_transformer_sharded():
    # Not using run_graph_test: it runs the model on CPU as a golden reference
    # first, and the 8B transformer CPU pass is far too slow. TT-only compile /
    # run smoke — one transformer forward = one denoise step for the raw DiT.
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": "1",
        }
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)

    compiled = torch.compile(model, backend="tt")

    # Hard-coded SP=4 x TP=8 hybrid sharding for a 32-chip Galaxy (throwaway
    # bringup config, NOT the loader's 1-D-Megatron get_mesh_config). Mesh axis
    # "model" (size 8) is tensor-parallel; axis "sp" (size 4) is sequence-parallel.
    mesh = get_mesh((4, 8), ("sp", "model"))

    # Weights: reuse the loader's TP spec. Its partition specs reference only the
    # "model" axis, so every weight is TP=8-sharded and left replicated across the
    # "sp" axis -- exactly what tensor-parallel weights want under SP.
    shard_spec = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Move inputs to device recursively: some transformer inputs (e.g. the
    # per-block SmolLM3 text_encoder_layers) arrive as a list of tensors, so a
    # top-level torch.is_tensor check leaves the nested tensors on CPU and the
    # first caption_projection matmul hits a cpu/xla device mismatch.
    def to_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return type(value)(to_device(v) for v in value)
        if isinstance(value, dict):
            return {k: to_device(v) for k, v in value.items()}
        return value

    inputs = tuple(to_device(value) for value in inputs)

    # Sequence parallelism (the "SP=4" half): shard the image latent stream
    # (hidden_states, shape (2, 4096, 48)) along its sequence dim over "sp";
    # 4096 / 4 = 1024. The text stream (seq 45) is not divisible by 4, so it is
    # left replicated across "sp" -- the standard SP split of long-image vs
    # short-text tokens. hidden_states is the only rank-3 input with seq 4096.
    for value in inputs:
        if torch.is_tensor(value) and value.dim() == 3 and value.shape[1] == 4096:
            xs.mark_sharding(value, mesh, (None, "sp", None))

    # Benchmark one denoising step (= one DiT forward). Run 6 iterations:
    # iteration 0 is a warmup that absorbs kernel JIT compilation and is
    # discarded; the following 5 are timed. We force each step to fully complete
    # before stopping the clock by materializing the output to host (.cpu()),
    # which blocks on the async TT device execution.
    NUM_WARMUP = 1
    NUM_TIMED = 5

    def run_denoising_step():
        out = compiled(*inputs)
        out.cpu()  # block until the device finishes this forward

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            run_denoising_step()

        timings_ms = []
        for _ in range(NUM_TIMED):
            start = time.perf_counter()
            run_denoising_step()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    mn = min(timings_ms)
    mx = max(timings_ms)
    avg = sum(timings_ms) / len(timings_ms)

    # Print the table before the process exits, since TT device teardown can
    # hang after the test body completes.
    print(
        f"\n===== FIBO DiT denoising-step benchmark (SP=4 x TP=8, Galaxy) =====\n"
        f"warmup (discarded): {NUM_WARMUP}    timed iterations: {NUM_TIMED}"
    )
    for i, t in enumerate(timings_ms, 1):
        print(f"  run {i}: {t:10.3f} ms")
    print(f"  {'-' * 24}")
    print(f"  {'min':<4}{mn:>12.3f} ms")
    print(f"  {'max':<4}{mx:>12.3f} ms")
    print(f"  {'avg':<4}{avg:>12.3f} ms")
    print("=" * 64)

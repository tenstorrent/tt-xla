# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for the FIBO (briaai/FIBO) text encoder — SmolLM3-3B.

FIBO is BRIA AI's 8B DiT text-to-image model; it conditions the DiT on the
*hidden states* of its text encoder, a base ``SmolLM3Model`` (the ``SmolLM3-3B``
causal LM with the vocab head discarded). So the text encoder runs as a single
forward that returns ``last_hidden_state`` — there is no autoregressive decode.
That makes the LLM harness (``test_llm``/``test_llm_tp``, which time prefill +
decode over a KV cache) the wrong contract, and the encoder harness
(``test_encoder``) is single-chip only. The encoder must run tensor-parallel:
model-bringup established a TP-4 baseline (mesh ``(1, 4)``) and it OOMs on a
single chip at long context — exactly the situation the Wan UMT5 text encoder is
in.

So, like the FIBO DiT perf test, this drives the generic sharded-component
harness ``benchmark_video_gen_torch_xla`` straight from the tt_forge_models
loader's public hooks (``load_model`` / ``load_inputs`` / ``get_mesh_config`` /
``load_shard_spec``): Megatron-1D (column→row) tensor parallelism over a
``(1, 4)`` mesh on a 4-chip Blackhole device (qb2), batch size 1.

Context length is pinned to 24576 — the largest context validated under TP-4
during model-bringup (PCC 0.9987). 32768 compiles but OOMs at runtime (the
materialized ``[1, 1, seq, seq]`` causal mask plus the O(seq^2) attention scores
exceed per-chip DRAM), and the architectural max (65536) needs more chips or a
mask-free attention path — so 24576 is the "max possible context on the TP-4
goal topology."

Reference: https://huggingface.co/briaai/FIBO
"""

import json
import os

import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from infra.utilities.torch_multichip_utils import get_mesh
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig

# Largest context validated on TP-4 (4x Blackhole, 32 GB/chip) during
# model-bringup; also the loader's DEFAULT_CONTEXT_LENGTH. Pin it here so the
# perf number is deterministic regardless of any FIBO_TE_CONTEXT_LENGTH in the
# CI env (32768 OOMs at runtime, 65536 needs more chips — see the module docs).
MAX_TP4_CONTEXT_LENGTH = 24576

# Bringup-safe compiler defaults; model-perf-tuning ramps these to headline perf.
COMPILER_CONFIG = CompilerConfig(optimization_level=0, enable_trace=False)

# Harness floor — model-bringup measured PCC 0.9987 at ctx 24576 under TP-4.
REQUIRED_PCC = 0.97


class FiboTextEncoderWrapper(torch.nn.Module):
    """Adapts SmolLM3Model to the ``wrapper(*inputs) -> tensor`` harness contract.

    The benchmark harness calls ``wrapper(*inputs)`` positionally and expects a
    single tensor, so this maps ``(input_ids, attention_mask)`` onto the model's
    kwargs forward and extracts ``last_hidden_state`` from its
    ``BaseModelOutputWithPast`` — the conditioning embedding FIBO's DiT consumes.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def test_fibo_text_encoder(output_file, request):
    from third_party.tt_forge_models.fibo.text_encoder.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Pin the context length to the TP-4 validated maximum (independent of any
    # external FIBO_TE_CONTEXT_LENGTH); load_inputs reads this env var.
    os.environ["FIBO_TE_CONTEXT_LENGTH"] = str(MAX_TP4_CONTEXT_LENGTH)

    loader = ModelLoader(variant=ModelVariant.BASE)
    model_info_name = loader.get_model_info().name
    print(f"\nLoading model {model_info_name}...")

    model = loader.load_model(dtype_override=torch.bfloat16)
    wrapper = FiboTextEncoderWrapper(model).eval()

    inputs = loader.load_inputs(batch_size=1)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Tensor-parallel plumbing from the loader's public hooks: the mesh from
    # get_mesh_config (TP-4 = (1, 4) on a 4-chip device) and the Megatron
    # column→row weight shard spec from load_shard_spec.
    def mesh_fn():
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        return get_mesh(mesh_shape, mesh_names)

    def apply_sharding_fn(wrapper_on_device, mesh):
        specs = loader.load_shard_spec(wrapper_on_device.model)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=[input_ids, attention_mask],
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=COMPILER_CONFIG,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        sharded=True,
        mesh_fn=mesh_fn,
        apply_sharding_fn=apply_sharding_fn,
        required_pcc=REQUIRED_PCC,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)

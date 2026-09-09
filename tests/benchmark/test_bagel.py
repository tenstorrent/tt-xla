# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark for the BAGEL ``bagel_model`` backbone.

``JiaxinGe/Diffusers-BAGEL`` is ByteDance's unified multimodal model, shipped as a
``trust_remote_code`` diffusers ``BagelPipeline``. The heavy component is
``bagel_model``: a Qwen2-7B-scale Mixture-of-Transformers (hidden 3584, intermediate
18944, 28 attention heads, 4 KV heads, 28 layers, ~27 GiB bf16). This benchmark
measures one forward through the understanding ("und") text path, which is the
component the functional test in ``tests/torch/models/bagel/`` brings up.

The backbone is weight-bound on a single n150/p150, so it runs **4-way tensor
parallel** (Megatron-1D, ``(None, "model")``) on the Wormhole fabric. 28 attention
heads is not divisible by 8, so 8-way TP is invalid; 4-way is the largest
head-divisible degree (28/4 = 7 q-heads, 4/4 = 1 kv-head per shard, so GQA stays
consistent) and it fits within ~12 GiB/chip.

**This is why the runner must expose exactly four chips.** Unlike every other
tensor-parallel loader here, BAGEL's ``get_mesh_config`` accepts only 1, 2 or 4 --
and ``get_mesh`` builds the mesh from *all* visible device ids, so a (1, 4) mesh
cannot be carved out of an 8-chip box; the run would die on the loader's
"Unsupported device count: 8".

The perf matrix has no per-entry ``env`` field, so the visible set is pinned here,
at import time, before torch-xla brings the plugin up. ``TT_VISIBLE_DEVICES``
selects **PCI cards, not chips** -- measured on an n300 llmbox, one card enumerates
two chips, so ``0,1`` yields 4 devices while ``0,1,2,3`` still yields 8. The pin
uses ``setdefault``, so an explicit ``TT_VISIBLE_DEVICES`` in the environment always
wins; override it on any runner whose chips-per-card differs.

The measurement goes through ``benchmarks/video_gen_benchmark.py`` -- despite the
name it is the only harness that supports SPMD sharding, so tensor-parallel
components land there regardless of modality (LongCat-Image does the same). It
builds a CPU golden, compiles for TT, runs warmup + timed iterations, and gates on
PCC, so a numerics regression fails the benchmark instead of quietly reporting a
fast wrong answer.

NOTE: the loader's forward is a single fixed-shape pass over the text backbone at
``DEFAULT_SEQ_LEN`` (128), batch 1 -- not autoregressive generation. The reported
figure is therefore per-forward latency and samples/sec, not TTFT or tokens/sec.
"""

import os

# Two n300 cards == 4 chips, the exact tensor-parallel degree this backbone needs.
# Must be set before torch-xla instantiates the TT plugin, hence above the imports
# below; ``setdefault`` leaves any operator-supplied value alone.
_DEFAULT_VISIBLE_CARDS = "0,1"
os.environ.setdefault("TT_VISIBLE_DEVICES", _DEFAULT_VISIBLE_CARDS)

import json

import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from infra.utilities.torch_multichip_utils import get_mesh
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.bagel.pytorch import ModelLoader, ModelVariant

SEED = 42
DATA_FORMAT = torch.bfloat16

# Compile options must stay in sync with the functional test in
# ``tests/torch/models/bagel/test_bagel_model.py`` so the PCC gate there and the
# perf number here compile the backbone identically. That test passes no
# compiler config, i.e. plain defaults; if one is ever added, change it in both
# places.
BACKBONE_COMPILER_CONFIG = CompilerConfig()

# Same gate as the functional test, which uses the infra ``PccConfig`` default
# (0.99). The benchmark must not report a number for a run the functional suite
# would fail.
REQUIRED_PCC = 0.99

MODEL_TYPE = "Multimodal, Single Forward, Random Input Data"

# Largest head-divisible tensor-parallel degree for this backbone (28 q-heads).
TP_DEGREE = 4


def _build_mesh():
    """Build the Megatron-1D mesh, insisting on exactly ``TP_DEGREE`` devices.

    ``get_mesh`` derives ``device_ids`` from the full visible device count, so the
    mesh shape has to consume every visible chip. Checking here turns an opaque
    loader ValueError (or a torch-xla mesh size assertion) into an actionable
    message.
    """
    num_devices = xr.global_runtime_device_count()
    if num_devices != TP_DEGREE:
        raise RuntimeError(
            f"BAGEL bagel_model needs exactly {TP_DEGREE} visible chips for "
            f"{TP_DEGREE}-way tensor parallelism, but the runtime reports "
            f"{num_devices}. 28 attention heads are not divisible by 8, so wider "
            f"degrees are invalid, and the mesh is built from all visible device "
            f"ids. This module pins TT_VISIBLE_DEVICES="
            f"{_DEFAULT_VISIBLE_CARDS!r} at import time, which gives 4 chips on an "
            f"n300 llmbox (the variable selects cards, not chips -- one n300 card "
            f"enumerates two chips). It is currently "
            f"{os.environ.get('TT_VISIBLE_DEVICES')!r}; set it to whichever cards "
            f"add up to {TP_DEGREE} chips on this runner."
        )

    mesh_shape, mesh_names = ModelLoader(ModelVariant.BAGEL_MODEL).get_mesh_config(
        num_devices
    )
    return get_mesh(mesh_shape, mesh_names)


def _sharding_fn(loader):
    """Mark the Megatron-1D weight specs on the on-device backbone.

    ``load_shard_spec`` maps each backbone parameter to a partition spec --
    column-parallel for q/k/v/gate/up, row-parallel for o_proj/down_proj,
    replicated for norms and embeddings. The harness calls this after the wrapper
    has been moved to the device, so the parameters walked here are already XLA
    tensors.
    """

    def apply_sharding(wrapper, mesh):
        specs = loader.load_shard_spec(wrapper)
        for tensor, spec in (specs or {}).items():
            xs.mark_sharding(tensor, mesh, spec)

    return apply_sharding


def test_bagel_model(output_file, request):
    """Qwen2 MoT backbone, text path -- one forward, 4-way tensor parallel."""
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.BAGEL_MODEL)
    # Do NOT call .eval() here, and do not "fix" this by adding it. The wrapper
    # deliberately holds the backbone in train mode: its forward drives
    # ``decoder_layer.forward_train`` explicitly, and inside that the attention
    # submodule is reached through ``PackedAttentionMoT.forward``, which dispatches
    # on ``self.training``. In eval mode that routes to ``forward_inference``
    # (flash-attn + KV cache), whose signature does not take ``packed_sequence``,
    # and the forward dies with a TypeError before any compilation happens.
    # Config dropout is 0.0, so train mode is numerically eval-equivalent here --
    # this is the same thing the functional test in tests/torch/models/bagel/ does.
    wrapper = loader.load_model(dtype_override=DATA_FORMAT)
    inputs = loader.load_inputs(seq_len=loader.DEFAULT_SEQ_LEN)

    model_info_name = "BAGEL-bagel_model"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running BAGEL benchmark: {model_info_name} (TP={TP_DEGREE})")

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=[inputs],
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=BACKBONE_COMPILER_CONFIG,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        sharded=True,
        mesh_fn=_build_mesh,
        apply_sharding_fn=_sharding_fn(loader),
        extract_output_tensor_fn=loader.unpack_forward_output,
        required_pcc=REQUIRED_PCC,
        model_type=MODEL_TYPE,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)

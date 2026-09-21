# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 — AVTransformer3DModel (the ~19B DiT) single-device component test.

The model is the audio-video diffusion transformer ported from the native
``ltx_core`` codebase (github.com/Lightricks/LTX-2). Two product tiers share
the SAME architecture and differ only in checkpoint weights:

    Fast -> ltx-2.3-22b-distilled-1.1.safetensors  (8-step distilled)
    Pro  -> ltx-2.3-22b-dev.safetensors            (full dev, two-stage HQ)

Runs one DiT forward and compares CPU vs TT output at PCC 0.99. The loader
builds the model from the checkpoint's embedded config with RANDOM weights
(no 46GB download) and wraps the ``Modality``-based forward in a plain-tensor
wrapper, so the harness can trace it.

NOTE: at full 48 layers this is ~19B params (~38GB bf16). It instantiates on
host RAM but is weight-bound on a single TT chip — this test is expected to
exercise the single-chip OOM/weight-bound path. The follow-up bringup is
multichip tensor-parallel (see loader.get_mesh_config / load_shard_spec).
"""

import os

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.ltx2_3.pytorch import ModelLoader, ModelVariant


def test_ltx2_3_transformer_fast():
    _run(ModelVariant.LTX2_3_FAST)


def test_ltx2_3_transformer_pro():
    _run(ModelVariant.LTX2_3_PRO)


@pytest.mark.xfail(
    reason="TT_FATAL @ tt_metal/impl/tensor/tensor_apis.cpp:631: buffers.size() == 1 — "
    "'Can't get a single buffer from host storage distributed over mesh shape "
    "MeshShape([1, 8])' in ccl::run(DistributeTensorOp), surfacing as "
    "'Bad StatusOr access: INTERNAL: Error code: 13'. The 48L graph compiles "
    "(30m54s, mesh (1,8) over 8 devices) but fatals at execution. NOTE: the status "
    "code matches tt-mlir#9014's title but this is a different cause (runtime fatal "
    "after a good compile, not ingest rejection) — do not reopen #9014."
)
def test_transformer_sharded():
    """LTX-2.3 AVTransformer3DModel (~19B DiT) tensor-parallel run on the TT mesh.

    Megatron-1D TP over the transformer blocks (see loader.get_mesh_config /
    load_shard_spec). Following the Mochi TP MVP, this is TT-only: the 19B CPU
    golden forward is prohibitively slow, so no PCC gate here — the goal is to
    prove the model compiles and runs on the multichip mesh. PCC re-enabled
    later via run_graph_test once the TT path is green.

    The body MUST materialize its output. Under lazy XLA, ending at
    ``compiled(*inputs)`` only *enqueues*: the body returns before the async
    device execution lands, so the test reported ``1 passed in 1854.61s``
    directly above a ``TT_FATAL`` with ``PYTEST_EXIT=1`` — green in CI on a
    model that never executed. ``torch_xla.sync()`` + ``.cpu()`` + a real
    assertion are what make the result mean anything. Do not remove them.
    """
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.LTX2_3_PRO)
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)

    compiled = torch.compile(model, backend="tt")

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    host_inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [t.to(device) for t in host_inputs]

    with torch.no_grad():
        video_out = compiled(*inputs)

    # Force the enqueued graph to actually execute and come back to host. A
    # device-side fatal surfaces here rather than after the test has passed.
    torch_xla.sync()
    video_out = video_out.cpu()

    # proj_out maps inner_dim -> out_channels, so batch and token count are
    # preserved from the video latent; the feature dim is not asserted.
    video_latent = host_inputs[0]
    assert video_out.shape[:2] == video_latent.shape[:2], (
        f"video output {tuple(video_out.shape)} does not preserve "
        f"(batch, tokens) of the video latent {tuple(video_latent.shape)}"
    )
    assert torch.isfinite(video_out).all(), "video output contains NaN/Inf"


def _run(variant: ModelVariant):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    loader = ModelLoader(variant)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )

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


def test_transformer_sharded():
    """LTX-2.3 AVTransformer3DModel (~19B DiT) tensor-parallel run on the TT mesh.

    Megatron-1D TP over the transformer blocks (see loader.get_mesh_config /
    load_shard_spec). Following the Mochi TP MVP, this is TT-only: the 19B CPU
    golden forward is prohibitively slow, so no PCC gate here — the goal is to
    prove the model compiles and runs on the multichip mesh. PCC re-enabled
    later via run_graph_test once the TT path is green.
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

    inputs = [t.to(device) for t in loader.load_inputs(dtype_override=torch.bfloat16)]

    with torch.no_grad():
        compiled(*inputs)


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

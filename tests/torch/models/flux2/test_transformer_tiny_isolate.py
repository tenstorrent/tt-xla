# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
THROWAWAY isolation harness for the FLUX.2 transformer sharded PCC≈0 failure.

Builds a *tiny* Flux2Transformer2DModel (1 dual + 1 single block) from the real
config — same heads(48)/head_dim(128)/dims, so the shard spec is exercised
identically — but small enough to fit UNSHARDED on a single chip. Random weights
+ random inputs; PCC measured TT-vs-CPU.

Matrix decomposes the failure:
  - unsharded bf16   : fails => MODEL-OP bug (sharding not involved)
  - sharded   bf16   : fails (but unsharded passed) => SHARD-SPEC bug
  - sharded   bfp8   : fails (but sharded bf16 passed) => bfp8 interaction
"""

import pytest
import torch
import torch_xla.runtime as xr
from diffusers import Flux2Transformer2DModel
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    GUIDANCE_SCALE,
    LATENT_GRID_H,
    LATENT_GRID_W,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    REPO_ID,
    Flux2TransformerWrapper,
    prepare_latent_image_ids,
    prepare_text_ids,
    shard_transformer_specs,
)

DTYPE = torch.bfloat16


def _build_tiny_model():
    config = Flux2Transformer2DModel.load_config(REPO_ID, subfolder="transformer")
    config = dict(config)
    config["num_layers"] = 1
    config["num_single_layers"] = 1
    torch.manual_seed(0)
    model = Flux2Transformer2DModel.from_config(config).to(DTYPE).eval()
    return Flux2TransformerWrapper(model).eval()


def _build_inputs():
    torch.manual_seed(1)
    inner_in = 128  # in_channels (packed latent channels)
    joint_dim = 15360  # encoder hidden (context_embedder input)
    seq_img = LATENT_GRID_H * LATENT_GRID_W  # 64 at 128x128
    hidden_states = torch.randn(1, seq_img, inner_in, dtype=DTYPE)
    encoder_hidden_states = torch.randn(1, MAX_SEQUENCE_LENGTH, joint_dim, dtype=DTYPE)
    timestep = torch.tensor([0.5], dtype=DTYPE)
    guidance = torch.tensor([GUIDANCE_SCALE], dtype=DTYPE)
    txt_ids = prepare_text_ids(1, MAX_SEQUENCE_LENGTH, DTYPE)
    img_ids = prepare_latent_image_ids(1, LATENT_GRID_H, LATENT_GRID_W, DTYPE)
    return [hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance]


def _pcc(a, b):
    a = a.detach().cpu().to(torch.float32).flatten()
    b = b.detach().cpu().to(torch.float32).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return float("nan")
    return float((a @ b).item() / denom)


def _make_comparator(tag):
    def _cmp(tt_res, cpu_res, args, kwargs):
        tt = tt_res[0] if isinstance(tt_res, (list, tuple)) else tt_res
        cpu = cpu_res[0] if isinstance(cpu_res, (list, tuple)) else cpu_res
        pcc = _pcc(tt, cpu)
        print(f"\n>>> TINY-ISOLATE [{tag}] PCC = {pcc:.6f}  "
              f"tt.shape={tuple(tt.shape)} cpu.shape={tuple(cpu.shape)}\n")
    return _cmp


@pytest.mark.parametrize(
    "tag,sharded,weight_dtype",
    [
        ("unsharded_bf16", False, ""),
        ("sharded_bf16", True, ""),
        ("sharded_bfp8", True, "bfp_bf8"),
    ],
)
def test_tiny(tag, sharded, weight_dtype):
    xr.set_device_type("TT")
    model = _build_tiny_model()
    inputs = _build_inputs()

    mesh = None
    shard_spec_fn = None
    compiler_config = None
    if sharded:
        n = xr.global_runtime_device_count()
        mesh = get_mesh((1, n), MESH_NAMES)
        shard_spec_fn = lambda m: shard_transformer_specs(m.transformer)
    if weight_dtype:
        compiler_config = CompilerConfig(experimental_weight_dtype=weight_dtype)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        custom_comparator=_make_comparator(tag),
    )

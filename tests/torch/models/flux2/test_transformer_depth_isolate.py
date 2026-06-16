# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
THROWAWAY depth-sweep harness for the FLUX.2 transformer sharded bfp8 PCC collapse.

Same as test_transformer_tiny_isolate but the block depth is configurable via env
(FLUX_NL dual blocks, FLUX_NS single blocks) and the weight dtype via FLUX_WDTYPE
("" = bf16, "bfp_bf8"). Always sharded across all devices. Lets us measure how PCC
decays with depth under bfp8 vs bf16 — to confirm the full-model collapse is bfp8
error accumulating over the 8+48 blocks (not a per-op / shard-spec bug).
"""

import os

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
NL = int(os.environ.get("FLUX_NL", "4"))
NS = int(os.environ.get("FLUX_NS", "20"))
WDTYPE = os.environ.get("FLUX_WDTYPE", "")


def _build_model():
    config = Flux2Transformer2DModel.load_config(REPO_ID, subfolder="transformer")
    config = dict(config)
    config["num_layers"] = NL
    config["num_single_layers"] = NS
    torch.manual_seed(0)
    model = Flux2Transformer2DModel.from_config(config).to(DTYPE).eval()
    return Flux2TransformerWrapper(model).eval()


def _build_inputs():
    torch.manual_seed(1)
    inner_in = 128
    joint_dim = 15360
    seq_img = LATENT_GRID_H * LATENT_GRID_W
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
        print(f"\n>>> DEPTH-ISOLATE [{tag}] PCC = {pcc:.6f}\n")

    return _cmp


def test_depth():
    xr.set_device_type("TT")
    model = _build_model()
    inputs = _build_inputs()
    n = xr.global_runtime_device_count()
    mesh = get_mesh((1, n), MESH_NAMES)
    shard_spec_fn = lambda m: shard_transformer_specs(m.transformer)
    compiler_config = CompilerConfig(experimental_weight_dtype=WDTYPE) if WDTYPE else None
    tag = f"nl{NL}_ns{NS}_{WDTYPE or 'bf16'}"
    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        custom_comparator=_make_comparator(tag),
    )

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
THROWAWAY isolation: REAL pretrained FLUX.2 transformer weights, truncated depth.

The random-weight depth sweep (test_transformer_depth_isolate) showed bfp8 == bf16
even at 24 blocks, so the full-model bfp8 collapse (pcc=-0.027) is NOT generic depth
accumulation. This harness loads the REAL pretrained transformer and keeps only the
first FLUX_NL dual + FLUX_NS single blocks, then compares bf16 vs bfp8 sharded — to
test whether the REAL weight distribution is what bfp8 cannot represent.

Env: FLUX_NL, FLUX_NS (block counts), FLUX_WDTYPE ("" bf16 | "bfp_bf8").
"""

import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    MESH_NAMES,
    shard_transformer_specs,
)

NL = int(os.environ.get("FLUX_NL", "4"))
NS = int(os.environ.get("FLUX_NS", "20"))
WDTYPE = os.environ.get("FLUX_WDTYPE", "")
SHARDED = os.environ.get("FLUX_SHARDED", "1") != "0"


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
        print(f"\n>>> REALWT-ISOLATE [{tag}] PCC = {_pcc(tt, cpu):.6f}\n")

    return _cmp


def test_realwt():
    xr.set_device_type("TT")
    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Truncate the REAL pretrained block stacks in place.
    t = model.transformer
    t.transformer_blocks = t.transformer_blocks[:NL]
    t.single_transformer_blocks = t.single_transformer_blocks[:NS]
    if hasattr(t, "config"):
        try:
            t.config.num_layers = NL
            t.config.num_single_layers = NS
        except Exception:
            pass

    mesh = None
    shard_spec_fn = None
    mode = os.environ.get("FLUX_SHARD_MODE", "all")  # all | blocks_only | mod_only
    if SHARDED:
        n = xr.global_runtime_device_count()
        mesh = get_mesh((1, n), MESH_NAMES)

        def _spec_fn(m):
            t = m.transformer
            specs = shard_transformer_specs(t)
            if mode == "all":
                return specs
            # Identify the "modulation / embedder / norm_out" (non per-block-matmul)
            # weights so we can selectively replicate them.
            mod_ids = set()
            for mod in (
                t.double_stream_modulation_img,
                t.double_stream_modulation_txt,
                t.single_stream_modulation,
            ):
                mod_ids.add(id(mod.linear.weight))
            mod_ids.add(id(t.x_embedder.weight))
            mod_ids.add(id(t.context_embedder.weight))
            for emb in (
                t.time_guidance_embed.timestep_embedder,
                t.time_guidance_embed.guidance_embedder,
            ):
                for ln in (emb.linear_1, emb.linear_2):
                    mod_ids.add(id(ln.weight))
                    if ln.bias is not None:
                        mod_ids.add(id(ln.bias))
            if hasattr(t.norm_out, "linear"):
                mod_ids.add(id(t.norm_out.linear.weight))
                if t.norm_out.linear.bias is not None:
                    mod_ids.add(id(t.norm_out.linear.bias))
            for mod in (
                t.double_stream_modulation_img,
                t.double_stream_modulation_txt,
                t.single_stream_modulation,
            ):
                if mod.linear.bias is not None:
                    mod_ids.add(id(mod.linear.bias))

            out = {}
            for w, s in specs.items():
                is_mod = id(w) in mod_ids
                if mode == "blocks_only" and is_mod:
                    s = tuple(None for _ in s)  # replicate the modulation group
                elif mode == "mod_only" and not is_mod:
                    s = tuple(None for _ in s)  # replicate everything else
                out[w] = s
            return out

        shard_spec_fn = _spec_fn
    compiler_config = (
        CompilerConfig(experimental_weight_dtype=WDTYPE) if WDTYPE else None
    )
    tag = f"real_nl{NL}_ns{NS}_{WDTYPE or 'bf16'}_{'sh' if SHARDED else 'unsh'}"
    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        custom_comparator=_make_comparator(tag),
    )

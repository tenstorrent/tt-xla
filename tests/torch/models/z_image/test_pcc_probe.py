# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TEMP PCC probe: localize the Z-Image sharded-transformer PCC drop (#5351).

After tt-mlir #8874 the sharded ZImageTransformer2DModel compiles & runs but the
30-layer model lands at PCC 0.626 (req 0.99) while unsharded passes. Per-layer
error compounds (2 layers ~0.99). Hypothesis: row-parallel all_reduce (attention
`to_out`, FFN `w2`) sums bf16 partials -> extra rounding the unsharded
single-fp32-matmul path never pays.

Probes (all on a truncated 2-layer DiT at latent 32x32 so they run fast):
  single         unsharded baseline
  model2         model axis size 2 (30 % 2 == 0, whole heads)
  model4         model axis size 4 (30 % 4 != 0, mid-head split)
  model4_fp32acc model4 + fp32_dest_acc_en (matmul accumulation only)
  model4_fp32    model4 in float32 everywhere (decisive precision test)
  no_attn_reduce model4 bf16 but `to_out` replicated -> all_gather, no reduce
  no_ffn_reduce  model4 bf16 but FFN `w2` replicated -> all_gather, no reduce
  no_reduce      model4 bf16 but both replicated -> no row-parallel reduce at all

Every probe logs PCC (never asserts) so passing configs report a number too.
"""
import os

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh

from tests.infra.testers.compiler_config import CompilerConfig
import third_party.tt_forge_models.z_image.pytorch.src.model_utils as mu
from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant

# shrink resolution (latent 32x32 -> short seq) so it fits small device counts
mu.LATENT_H, mu.LATENT_W = 32, 32
# Truncate the 30-layer DiT. Default 2 keeps the unsharded baseline on one chip;
# raise via ZIMAGE_NLAYERS to amplify per-layer error compounding on the sharded
# variants so the hypotheses separate cleanly.
NLAYERS = int(os.environ.get("ZIMAGE_NLAYERS", "2"))


def _pcc(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return float("nan") if denom == 0 else float((vx @ vy) / denom)


def _logging_comparator(label):
    """Return a custom_comparator that prints PCC for each output leaf and never raises."""

    def cmp(tt_res, cpu_res, args, kwargs):
        tt = tt_res if isinstance(tt_res, (list, tuple)) else [tt_res]
        cpu = cpu_res if isinstance(cpu_res, (list, tuple)) else [cpu_res]
        pccs = []
        for t, c in zip(tt, cpu):
            if isinstance(t, torch.Tensor) and isinstance(c, torch.Tensor):
                pccs.append(_pcc(t.cpu(), c.cpu()))
        worst = min(pccs) if pccs else float("nan")
        print(f"\n[PCC-PROBE] {label}: pcc={worst:.6f}  (all={[round(p, 6) for p in pccs]})\n")

    return cmp


def _load_small(dtype=torch.bfloat16):
    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=dtype)
    t = model.transformer
    t.layers = t.layers[:NLAYERS]  # keep RoPE/attention; drop most blocks
    return loader, model


# --- selective-sharding shard_spec builders ---------------------------------
def _iter_blocks(transformer):
    yield from transformer.noise_refiner
    yield from transformer.context_refiner
    yield from transformer.layers


def _replicate_attention(model):
    """Replicate the ENTIRE attention of every block (q/k/v/to_out/qk-norm).

    Removes head-sharding, the RoPE all_to_all reshard (#8874 path) and the
    sharded SDPA. FFN stays Megatron-sharded. If PCC recovers, attention
    sharding (RoPE reshard / sharded SDPA) is the culprit, not FFN.
    """
    specs = mu.shard_transformer_specs(model.transformer)
    for block in _iter_blocks(model.transformer):
        a = block.attention
        for name in ("to_q", "to_k", "to_v"):
            proj = getattr(a, name)
            specs[proj.weight] = mu._replicate_spec(proj.weight)
            if proj.bias is not None:
                specs[proj.bias] = mu._replicate_spec(proj.bias)
        to_out = (
            a.to_out[0]
            if isinstance(a.to_out, (torch.nn.Sequential, torch.nn.ModuleList))
            else a.to_out
        )
        specs[to_out.weight] = mu._replicate_spec(to_out.weight)
        if to_out.bias is not None:
            specs[to_out.bias] = mu._replicate_spec(to_out.bias)
        for nm in ("norm_q", "norm_k"):
            n = getattr(a, nm, None)
            if n is not None and hasattr(n, "weight"):
                specs[n.weight] = mu._replicate_spec(n.weight)
    return specs


def _replicate_ffn(model):
    """Replicate the ENTIRE FFN of every block (w1/w2/w3). Attention stays sharded."""
    specs = mu.shard_transformer_specs(model.transformer)
    for block in _iter_blocks(model.transformer):
        ff = block.feed_forward
        for name in ("w1", "w2", "w3"):
            w = getattr(ff, name).weight
            specs[w] = mu._replicate_spec(w)
    return specs


def _replicate_embedders(model):
    """Replicate everything that is NOT a per-block attention/FFN weight:
    all_x_embedder, t_embedder.mlp, cap_embedder, final_layer (linear + adaLN)
    and every block's adaLN_modulation. Block attention + FFN stay sharded.

    Embedders + adaLN are the only params sharded in *every* failing config
    (repl_attn / repl_ffn both leave them sharded), so they are the prime
    remaining suspect for the dtype-independent structural drop.
    """
    t = model.transformer
    specs = mu.shard_transformer_specs(t)

    def repl(p):
        if p is not None:
            specs[p] = mu._replicate_spec(p)

    for embedder in t.all_x_embedder.values():
        repl(embedder.weight)
        repl(getattr(embedder, "bias", None))
    for fl in t.all_final_layer.values():
        repl(fl.linear.weight)
        repl(getattr(fl.linear, "bias", None))
        repl(fl.adaLN_modulation[1].weight)
        repl(getattr(fl.adaLN_modulation[1], "bias", None))
    repl(t.t_embedder.mlp[0].weight)
    repl(getattr(t.t_embedder.mlp[0], "bias", None))
    repl(t.t_embedder.mlp[2].weight)
    repl(getattr(t.t_embedder.mlp[2], "bias", None))
    repl(t.cap_embedder[1].weight)
    repl(getattr(t.cap_embedder[1], "bias", None))
    for block in _iter_blocks(t):
        if hasattr(block, "adaLN_modulation"):
            repl(block.adaLN_modulation[0].weight)
            repl(getattr(block.adaLN_modulation[0], "bias", None))
    return specs


def _column_shard_adaln(model):
    """Restore the OLD (buggy) column-sharding of adaLN on top of the now-fixed
    default spec. Used to test whether the 30-layer ttnn.concat L1-overflow is
    caused by adaLN replication or is pre-existing."""
    t = model.transformer
    specs = mu.shard_transformer_specs(t)
    for block in _iter_blocks(t):
        if hasattr(block, "adaLN_modulation"):
            lin = block.adaLN_modulation[0]
            specs[lin.weight] = ("model", None)
            if lin.bias is not None:
                specs[lin.bias] = ("model",)
    for fl in t.all_final_layer.values():
        ada = fl.adaLN_modulation[1]
        specs[ada.weight] = ("model", None)
        if ada.bias is not None:
            specs[ada.bias] = ("model",)
    return specs


def _replicate_adaln_only(model):
    """Replicate ONLY adaLN_modulation (per-block + final layer). Everything
    else (embedders + attention + FFN) stays sharded. Tests whether the
    chunk-along-sharded-dim adaLN is the dominant per-layer bug."""
    t = model.transformer
    specs = mu.shard_transformer_specs(t)

    def repl(p):
        if p is not None:
            specs[p] = mu._replicate_spec(p)

    for block in _iter_blocks(t):
        if hasattr(block, "adaLN_modulation"):
            repl(block.adaLN_modulation[0].weight)
            repl(getattr(block.adaLN_modulation[0], "bias", None))
    for fl in t.all_final_layer.values():
        repl(fl.adaLN_modulation[1].weight)
        repl(getattr(fl.adaLN_modulation[1], "bias", None))
    return specs


def _replicate_emb_only(model):
    """Replicate ONLY the embedders (all_x_embedder, t_embedder, cap_embedder).
    adaLN + attention + FFN stay sharded."""
    t = model.transformer
    specs = mu.shard_transformer_specs(t)

    def repl(p):
        if p is not None:
            specs[p] = mu._replicate_spec(p)

    for embedder in t.all_x_embedder.values():
        repl(embedder.weight)
        repl(getattr(embedder, "bias", None))
    repl(t.t_embedder.mlp[0].weight)
    repl(getattr(t.t_embedder.mlp[0], "bias", None))
    repl(t.t_embedder.mlp[2].weight)
    repl(getattr(t.t_embedder.mlp[2], "bias", None))
    repl(t.cap_embedder[1].weight)
    repl(getattr(t.cap_embedder[1], "bias", None))
    return specs


def _replicate_row_parallel(model, *, attn, ffn):
    """Base Megatron specs, but replicate chosen row-parallel weights.

    Replicating a row-parallel weight makes Shardy all_gather the sharded
    activation and run one full fp32-accumulated matmul (rounded once) instead
    of summing bf16 partials in an all_reduce.
    """
    specs = mu.shard_transformer_specs(model.transformer)
    for block in _iter_blocks(model.transformer):
        if attn:
            a = block.attention
            to_out = (
                a.to_out[0]
                if isinstance(a.to_out, (torch.nn.Sequential, torch.nn.ModuleList))
                else a.to_out
            )
            specs[to_out.weight] = (None, None)
        if ffn:
            specs[block.feed_forward.w2.weight] = (None, None)
    return specs


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model2_2dev_fullres():
    """model=2 (head-divisible, 30/2=15 whole heads) at FULL resolution, with the
    SAME per-chip footprint as a real 2-device deployment.

    SPMD requires the mesh to span all opened devices (8 here), so a literal
    (1,2) 2-device mesh is rejected. Instead use mesh (4,2) -> batch=4, model=2:
    the model is sharded 2-way (half the ~6.2B model per chip) and the batch axis
    only replicates, so per-chip memory (half model + full-res activations) is
    identical to a real 2-chip (1,2) model=2 mesh. Answers the OOM question and
    gives the full-res model=2 PCC."""
    xr.set_device_type("TT")
    torch.manual_seed(mu.SEED)
    # full resolution (override the module-level reduced 32x32)
    mu.LATENT_H, mu.LATENT_W = mu.latent_hw_from_pixels()
    n = xr.global_runtime_device_count()
    loader, model = _load_small()  # NLAYERS via ZIMAGE_NLAYERS (set 30 to run full)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    # Physical topology is 2x4, so keep shape (2,4); name the size-2 axis "model"
    # => model=2 (half model per chip == 2-device footprint), batch=4 replicates.
    mesh = Mesh(np.array(range(n)), (2, 4), ("model", "batch"))
    run_graph_test(
        model, inputs, framework=Framework.TORCH, mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
        custom_comparator=_logging_comparator("model2_2dev_fullres"),
    )


def _run(label, mesh_names, *, dtype=torch.bfloat16, compiler_config=None, shard_spec_fn=None):
    xr.set_device_type("TT")
    torch.manual_seed(mu.SEED)
    n = xr.global_runtime_device_count()
    loader, model = _load_small(dtype)
    inputs = loader.load_inputs(dtype_override=dtype)
    mesh = Mesh(np.array(range(n)), (2, 4), mesh_names)
    run_graph_test(
        model, inputs, framework=Framework.TORCH, mesh=mesh,
        shard_spec_fn=shard_spec_fn or loader.load_shard_spec,
        compiler_config=compiler_config,
        custom_comparator=_logging_comparator(label),
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_pcc_single():  # unsharded baseline (truncated model fits one chip)
    xr.set_device_type("TT")
    torch.manual_seed(mu.SEED)
    loader, model = _load_small()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    run_graph_test(model, inputs, framework=Framework.TORCH,
                   custom_comparator=_logging_comparator("single"))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model2():  # model axis = first (size 2); 30 % 2 == 0 -> whole heads
    _run("model2", ("model", "batch"))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4():  # model axis = second (size 4); 30 % 4 != 0 -> mid-head split
    _run("model4", ("batch", "model"))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4_fp32acc():  # fp32 accumulation in matmuls (not in the CCL)
    _run("model4_fp32acc", ("batch", "model"), compiler_config=CompilerConfig(fp32_dest_acc_en=True))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4_fp32():  # decisive: float32 everywhere incl. the all_reduce
    _run("model4_fp32", ("batch", "model"), dtype=torch.float32)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_no_attn_reduce():  # replicate attention to_out -> all_gather, no reduce
    def spec_fn(model):
        return _replicate_row_parallel(model, attn=True, ffn=False)

    _run("no_attn_reduce", ("batch", "model"), shard_spec_fn=spec_fn)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_no_ffn_reduce():  # replicate FFN w2 -> all_gather, no reduce
    def spec_fn(model):
        return _replicate_row_parallel(model, attn=False, ffn=True)

    _run("no_ffn_reduce", ("batch", "model"), shard_spec_fn=spec_fn)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_no_reduce():  # replicate both row-parallel weights -> no reduce anywhere
    def spec_fn(model):
        return _replicate_row_parallel(model, attn=True, ffn=True)

    _run("no_reduce", ("batch", "model"), shard_spec_fn=spec_fn)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_repl_attn():  # attention fully replicated (no RoPE reshard / sharded SDPA)
    _run("repl_attn", ("batch", "model"), shard_spec_fn=_replicate_attention)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_repl_ffn():  # FFN fully replicated; attention stays sharded
    _run("repl_ffn", ("batch", "model"), shard_spec_fn=_replicate_ffn)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model2_fp32():  # whole-head split in float32 (compiles; precision probe)
    _run("model2_fp32", ("model", "batch"), dtype=torch.float32)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_replicated():  # full replication on the mesh -> sanity: should be ~0.99
    _run("replicated", ("batch", "model"), shard_spec_fn=lambda model: {})


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_repl_embed():  # replicate embedders + adaLN; only blocks sharded
    _run("repl_embed", ("batch", "model"), shard_spec_fn=_replicate_embedders)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_repl_adaln_only():  # replicate ONLY adaLN_modulation
    _run("repl_adaln_only", ("batch", "model"), shard_spec_fn=_replicate_adaln_only)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_repl_emb_only():  # replicate ONLY the embedders
    _run("repl_emb_only", ("batch", "model"), shard_spec_fn=_replicate_emb_only)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_oldadaln():  # OLD column-sharded adaLN (diagnose 30-layer concat L1)
    _run("oldadaln", ("batch", "model"), shard_spec_fn=_column_shard_adaln)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4_opt2():  # fixed spec + opt level 2 (memory layout) to fit the concat
    _run("model4_opt2", ("batch", "model"),
         compiler_config=CompilerConfig(optimization_level=2))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_pcc_model4_opt1():  # fixed spec + opt level 1
    _run("model4_opt1", ("batch", "model"),
         compiler_config=CompilerConfig(optimization_level=1))

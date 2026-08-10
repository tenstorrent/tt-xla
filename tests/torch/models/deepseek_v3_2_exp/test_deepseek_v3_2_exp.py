# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.testers.compiler_config import CompilerConfig
from safetensors.torch import load_file as safetensors_load_file
from torch_xla.distributed.spmd import Mesh
from transformers import PreTrainedTokenizerFast
from tt_torch.custom_ops import (
    dsa_kernels_available,
    topk_large_indices_mask_invalid_slots,
)
from tt_torch.sharding import sharding_constraint_tensor
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.loader import (
    ModelLoader as DeepSeekV32ModelLoader,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.src.modified_model import (
    LayerNorm,
    ModelArgs,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.src.modified_model import (
    Transformer as ModifiedTransformer,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.src.modified_model import (
    apply_rotary_emb,
)

sys.path.insert(0, os.path.dirname(__file__))
from build_weight_cache import _dequant_cache_dir, _has_cache, build_cache

DEEPSEEK_V3_2_EXP_REPO = "deepseek-ai/DeepSeek-V3.2-Exp"


def _fix_layernorm_dtype(model):
    # LayerNorm calls x.float() internally and errors on mixed dtype, so
    # restore fp32 params that .to(bfloat16) silently converted.
    for module in model.modules():
        if isinstance(module, LayerNorm):
            module.weight.data = module.weight.data.to(torch.float32)
            module.bias.data = module.bias.data.to(torch.float32)


# This model is modified from the original deepseek_v3_2_exp model.py. Comments about each modification made can be found in
# third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/src/modified_model.py. Some of the notable modifications include:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
#    - fast_hadamard_transform requires a CUDA enviroment and fails to install
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
#    - the original implementation (kernel.py) relies on custom tilelang kernels not supported on TT
# 3. Avoid torch.view_as_complex/view_as_real operations


@pytest.mark.xfail(
    reason="TT_THROW: Statically allocated circular buffers on core range [(x=7,y=6) - (x=7,y=6)] grow to 16897152 B which is beyond max L1 size of 1499136 B"
)
def test_deepseek_modified_transformer_single_layer():
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    model = ModifiedTransformer(args)

    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)

    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.to("cpu")


def test_deepseek_complex_rotary_emb():
    xr.set_device_type("TT")

    # apply_rotary_emb function copied from model.py
    def apply_rotary_emb(
        x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
    ) -> torch.Tensor:
        dtype = x.dtype
        shape = x.shape
        if not interleaved:
            x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
        x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        if not interleaved:
            y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
        return y.to(dtype)

    batch_size = 2
    seq_len = 16
    dim = 64
    n_heads = 4
    head_dim = dim // n_heads

    x = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    freqs_cis = torch.randn(seq_len, head_dim // 2, dtype=torch.complex64)

    run_graph_test(
        apply_rotary_emb,
        [x, freqs_cis],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.lb_blackhole
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_attention_prefill(batch_size):
    xr.set_device_type("TT")
    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Prefill branch expects mask shape (bsz, seqlen, seqlen) for index_mask += mask
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)

    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(attention, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        # Conditionally shard weights that involve batch axis
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        shard_specs[args[0]] = (None, None, batch_axis)  # hidden_states
        shard_specs[args[3]] = (batch_axis, None, None)  # attention_mask
        shard_specs[attention.wq_b.weight] = ("model", None)
        shard_specs[attention.wkv_b.weight] = ("model", None)
        shard_specs[attention.wo.weight] = (batch_axis, "model")

        shard_specs[attention.wq_a.weight] = (None, batch_axis)
        shard_specs[attention.wkv_a.weight] = (None, batch_axis)

        shard_specs[attention.kv_cache] = (batch_axis, None, None)
        shard_specs[attention.pe_cache] = (batch_axis, None, None)

        # Indexer sharding
        shard_specs[attention.indexer.wq_b.weight] = ("model", None)
        shard_specs[attention.indexer.wk.weight] = (None, batch_axis)
        shard_specs[attention.indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[attention.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [
            hidden_states,  # input tensor
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.bh_galaxy
def test_deepseek_attention_prefill_bh_galaxy():
    """Same isolated-MLA-attention PCC check as test_deepseek_attention_prefill,
    but on the exact 32-device Galaxy mesh/shard-spec conventions used by the
    passing test_deepseek_v3_2_exp_tp_galaxy_2_layers benchmark (PCC~0.922 at
    2 real dense layers, no quantization) -- isolating whether attention alone
    already accounts for that near-threshold PCC, or whether it's clean here
    and the drift comes from elsewhere (indexer, RMSNorm, RoPE, or something
    only visible once several layers are stacked).

    mesh_shape=(4, 8) with axis names ("batch", "model") matches
    third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/loader.py's
    get_mesh_config(32) exactly (the real benchmark's mesh). Weight shard specs
    below are copied from that same loader's load_shard_spec -- model axis
    (size 8) carries ALL of wq_a/wkv_a/wq_b/wkv_b/wo (both projection
    directions), not just the output-feature axis the way the existing (2,4)
    unit test above splits it; batch axis (size 4) is reserved for the
    kv_cache/pe_cache request-batch dimension. This is a materially different
    sharding convention from test_deepseek_attention_prefill, not just a bigger
    mesh -- intentionally, to match what the benchmark actually exercised.

    Still uses the plain (unwrapped) ModifiedTransformer with random weights,
    like the (2,4) test above -- NOT the loader's DeepSeekV32ForCausalLM +
    MLACache/DeepseekV32IndexerCache wrapper the benchmark's model_name path
    uses, so kv_cache/pe_cache/indexer.k_cache keep their plain 3D buffer shape
    (batch, seq, head_dim) and shard spec here, not the wrapper's extra cache
    dimension. If this passes cleanly, that's still a real signal about the
    core attention math on this exact mesh; it does not by itself rule out the
    cache-wrapper plumbing as a separate contributor.

    No comparison_config is passed, so this uses run_graph_test's strict
    default (PCC >= 0.99) -- deliberately stricter than the 0.92 the full
    2-layer benchmark just barely cleared, so a failure's exception message
    reports the actual measured PCC rather than a coarse pass/fail at 0.95.
    """
    xr.set_device_type("TT")
    batch_size = 1
    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(attention, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        shard_specs[args[0]] = (batch_axis, None, None)  # hidden_states
        shard_specs[args[3]] = (batch_axis, None, None)  # attention_mask

        # Weight shard specs match loader.py::load_shard_spec exactly (the
        # real Galaxy benchmark's convention): "model" axis carries both
        # input- and output-feature sharding for every attention projection.
        shard_specs[attention.wq_a.weight] = (None, "model")
        shard_specs[attention.wkv_a.weight] = (None, "model")
        shard_specs[attention.wq_b.weight] = ("model", None)
        shard_specs[attention.wkv_b.weight] = ("model", None)
        shard_specs[attention.wo.weight] = (None, "model")

        shard_specs[attention.kv_cache] = (batch_axis, None, None)
        shard_specs[attention.pe_cache] = (batch_axis, None, None)

        # Indexer sharding (same loader.py convention)
        shard_specs[attention.indexer.wq_b.weight] = ("model", None)
        shard_specs[attention.indexer.wk.weight] = (None, "model")
        shard_specs[attention.indexer.weights_proj.weight] = (None, "model")
        shard_specs[attention.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    run_graph_test(
        attention,
        [
            hidden_states,  # input tensor
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.lb_blackhole
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_indexer(batch_size):
    xr.set_device_type("TT")

    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    indexer = model.layers[0].attn.indexer

    # Enable raw score return for testing (returns index_score instead of topk_indices)
    indexer.return_raw_scores = True

    # Create inputs
    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    qr = torch.randn((batch_size, seq_len, args.q_lora_rank), dtype=torch.bfloat16)
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    # Setup mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(indexer, args, kwargs):
        # Conditionally shard weights that involve batch axis
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        # Input tensors
        # hidden_states (x): [batch, seq, dim]
        shard_specs[args[0]] = (None, None, batch_axis)
        # qr: [batch, seq, q_lora_rank]
        shard_specs[args[1]] = (batch_axis, None, None)
        # attention_mask: [batch, seq, seq]
        shard_specs[args[4]] = (batch_axis, None, None)

        # Weight tensors
        # [n_heads*head_dim, q_lora_rank]
        shard_specs[indexer.wq_b.weight] = ("model", None)
        shard_specs[indexer.wk.weight] = (None, batch_axis)  # [head_dim, dim]
        shard_specs[indexer.k_norm.weight] = (None,)  # [head_dim]
        shard_specs[indexer.k_norm.bias] = (None,)  # [head_dim]
        # [n_heads, dim]
        shard_specs[indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[indexer.haddamard] = (None, None)  # [head_dim, head_dim]

        # Cache tensors
        # [max_batch, max_seq, head_dim]
        shard_specs[indexer.k_cache] = (batch_axis, None, None)

        # k_scale_cache if present (for FP8 quantization mode)
        if hasattr(indexer, "k_scale_cache"):
            shard_specs[indexer.k_scale_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        indexer,
        [
            hidden_states,
            qr,
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.bh_galaxy
def test_deepseek_indexer_bh_galaxy():
    """Same isolated-indexer PCC check as test_deepseek_indexer, but on the
    exact 32-device Galaxy mesh/shard-spec conventions used by the passing
    test_deepseek_v3_2_exp_tp_galaxy_2_layers benchmark. See
    test_deepseek_attention_prefill_bh_galaxy's docstring for the full
    rationale (mesh_shape=(4,8), loader.py::load_shard_spec conventions,
    plain ModifiedTransformer not the loader's cache-wrapper class, strict
    default PCC>=0.99 to surface the real measured value).
    """
    xr.set_device_type("TT")
    batch_size = 1
    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    indexer = model.layers[0].attn.indexer
    indexer.return_raw_scores = True

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    qr = torch.randn((batch_size, seq_len, args.q_lora_rank), dtype=torch.bfloat16)
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(indexer, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        shard_specs[args[0]] = (batch_axis, None, None)  # hidden_states
        shard_specs[args[1]] = (batch_axis, None, None)  # qr
        shard_specs[args[4]] = (batch_axis, None, None)  # attention_mask

        # Weight shard specs match loader.py::load_shard_spec exactly.
        shard_specs[indexer.wq_b.weight] = ("model", None)
        shard_specs[indexer.wk.weight] = (None, "model")
        shard_specs[indexer.k_norm.weight] = (None,)
        shard_specs[indexer.k_norm.bias] = (None,)
        shard_specs[indexer.weights_proj.weight] = (None, "model")
        shard_specs[indexer.haddamard] = (None, None)

        shard_specs[indexer.k_cache] = (batch_axis, None, None)
        if hasattr(indexer, "k_scale_cache"):
            shard_specs[indexer.k_scale_cache] = (batch_axis, None, None)

        return shard_specs

    run_graph_test(
        indexer,
        [
            hidden_states,
            qr,
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_deepseek_v3_2_moe_block(batch_size, seq_len):
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    repo_id = DEEPSEEK_V3_2_EXP_REPO

    # V3.2-Exp config has first_k_dense_replace=3; build 4 layers so HF layer 3
    # (first MoE layer) maps cleanly to model.layers[3].
    loader = DeepSeekV32ModelLoader(num_layers=4, max_batch_size=batch_size)
    loader._load_config(use_mla_cache=False, max_seq_len=seq_len * 2)
    args = loader._args
    # MoE-only test — skip indexer build; its cache weights become Unexpected at load.
    args.index_n_heads = 0

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)

    cache_dir = _dequant_cache_dir(repo_id, args.n_layers)
    if not _has_cache(cache_dir):
        build_cache(repo_id, args.n_layers, args.n_dense_layers)
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".safetensors"):
            state_dict.update(safetensors_load_file(os.path.join(cache_dir, fname)))
    model.load_state_dict(state_dict, strict=False)

    block = model.layers[args.n_dense_layers]

    mesh_shape = (2, 4)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)

    ffn = block.ffn
    ffn.eval()

    # AutoTokenizer.from_pretrained internally loads model config to determine tokenizer
    # class, which triggers a transformers 5.5 rope_scaling/max_position_embeddings bug
    # for unregistered model types (deepseek_v32). PreTrainedTokenizerFast loads only
    # tokenizer.json without touching model config.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id, padding_side="right")
    encoded = tokenizer(
        "Tell me a short story.",
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens = encoded["input_ids"][:, :seq_len].repeat(batch_size, 1)
    with torch.no_grad():
        hidden_states = model.embed(tokens).to(torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(ffn, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        mlp = ffn.mlp
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)

        shared = ffn.shared_experts
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.985),
    )

    run_graph_test(
        ffn,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.bh_galaxy
def test_deepseek_v3_2_dense_block_bh_galaxy():
    """Full single-layer Block (attn + indexer + dense FFN + both RMSNorms +
    residual) for layer 0 -- dense, since first_k_dense_replace=3 -- using
    REAL dequantized checkpoint weights, on the exact 32-device Galaxy
    mesh/shard-spec conventions test_deepseek_v3_2_exp_tp_galaxy_2_layers
    (the passing 2-layer benchmark, PCC~0.922) actually uses.

    test_deepseek_attention_prefill_bh_galaxy and test_deepseek_indexer_bh_galaxy
    both passed cleanly (PCC>=0.99) with RANDOM weights, isolating attention and
    the indexer individually. This test controls for what those two couldn't:
    the dense FFN, real (checkpoint-derived) weight value distributions instead
    of random noise, and whichever integration effects (attn+ffn+residual
    composed in one forward pass, matching Block.forward's pre-norm
    fused-residual pattern) only show up once a full real layer runs end to
    end. If this degrades notably below ~0.99, the problem is localized to
    layer 0 alone (FFN, integration, or real-weight sensitivity) with no MoE
    or cross-layer stacking required to reproduce it. If it stays clean, that
    points instead at cross-layer residual accumulation (layer 0 -> layer 1)
    as the next thing to check.

    Real-weight loading mirrors test_deepseek_v3_2_moe_block's pattern
    (DeepSeekV32ModelLoader + build_weight_cache.build_cache), but for
    num_layers=1 (only need layer 0) and WITHOUT zeroing index_n_heads --
    unlike the MoE-block test, this one wants the indexer built and its real
    checkpoint weights loaded, since the indexer is part of attention at every
    layer (dense or MoE) and the passing benchmark had it active
    (use_indexer_cache=True).

    Weight shard specs (attn/indexer same as the two component tests above,
    dense ffn.w1/w3 on ("batch","model") jointly, ffn.w2 on ("model","batch"),
    both RMSNorms on ("model",)) are copied verbatim from
    loader.py::load_shard_spec's per-layer entries.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    repo_id = DEEPSEEK_V3_2_EXP_REPO
    batch_size = 1
    seq_len = 32

    # num_layers=1 would trigger the loader's own clamp (n_dense_layers is
    # forced to max(0, num_layers-1)=0 whenever the real first_k_dense_replace
    # (3) is >= num_layers, so a 1-layer request deliberately gets a MoE
    # layer instead of dense). num_layers=2 clamps to n_dense_layers=1
    # instead, keeping layer 0 dense as intended, on a fresh cache path.
    loader = DeepSeekV32ModelLoader(num_layers=2, max_batch_size=batch_size)
    loader._load_config(use_mla_cache=False, max_seq_len=seq_len * 2)
    args = loader._args

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)

    cache_dir = _dequant_cache_dir(repo_id, args.n_layers)
    if not _has_cache(cache_dir):
        build_cache(repo_id, args.n_layers, args.n_dense_layers)
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".safetensors"):
            state_dict.update(safetensors_load_file(os.path.join(cache_dir, fname)))
    model.load_state_dict(state_dict, strict=False)

    block = model.layers[0]
    block.eval()

    # AutoTokenizer.from_pretrained internally loads model config to determine tokenizer
    # class, which triggers a transformers 5.5 rope_scaling/max_position_embeddings bug
    # for unregistered model types (deepseek_v32). PreTrainedTokenizerFast loads only
    # tokenizer.json without touching model config.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id, padding_side="right")
    encoded = tokenizer(
        "Tell me a short story.",
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens = encoded["input_ids"][:, :seq_len].repeat(batch_size, 1)
    with torch.no_grad():
        hidden_states = model.embed(tokens).to(torch.bfloat16)

    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(block, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        shard_specs[args[0]] = (batch_axis, None, None)  # hidden_states
        # args[1] is residual=None -- not a tensor, nothing to shard
        shard_specs[args[4]] = (batch_axis, None, None)  # attention_mask

        attn = block.attn
        shard_specs[attn.wq_a.weight] = (None, "model")
        shard_specs[attn.wkv_a.weight] = (None, "model")
        shard_specs[attn.wq_b.weight] = ("model", None)
        shard_specs[attn.wkv_b.weight] = ("model", None)
        shard_specs[attn.wo.weight] = (None, "model")
        shard_specs[attn.kv_cache] = (batch_axis, None, None)
        shard_specs[attn.pe_cache] = (batch_axis, None, None)

        idx = attn.indexer
        shard_specs[idx.wq_b.weight] = ("model", None)
        shard_specs[idx.wk.weight] = (None, "model")
        shard_specs[idx.weights_proj.weight] = (None, "model")
        shard_specs[idx.k_cache] = (batch_axis, None, None)

        ffn = block.ffn
        shard_specs[ffn.w1.weight] = (batch_axis, "model")
        shard_specs[ffn.w3.weight] = (batch_axis, "model")
        shard_specs[ffn.w2.weight] = ("model", batch_axis)

        shard_specs[block.attn_norm.weight] = ("model",)
        shard_specs[block.ffn_norm.weight] = ("model",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        block,
        [
            hidden_states,  # x
            None,  # residual (first layer)
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


class _TwoLayerStack(nn.Module):
    """Chains two Blocks, threading the pre-norm residual accumulator between
    them the way Transformer.forward does (Block.forward returns (x, residual)
    and expects residual=None only for the very first layer)."""

    def __init__(self, layer0, layer1):
        super().__init__()
        self.layer0 = layer0
        self.layer1 = layer1

    def forward(self, x, start_pos, freqs_cis, mask):
        x, residual = self.layer0(x, None, start_pos, freqs_cis, mask)
        x, residual = self.layer1(x, residual, start_pos, freqs_cis, mask)
        return x, residual


@pytest.mark.bh_galaxy
def test_deepseek_v3_2_dense_moe_2layer_transformer_bh_galaxy():
    """2-layer (dense + MoE) PCC check, isolating cross-layer residual
    accumulation as the difference between two earlier results:

      - test_deepseek_v3_2_dense_block_bh_galaxy (real weights, ONE dense
        layer, full attn+ffn+norms+residual) passed cleanly at PCC>=0.99.
      - test_deepseek_v3_2_exp_tp_galaxy_2_layers (the benchmark) passed only
        at PCC~0.922, with random weights and, per loader.py::_load_config's
        own clamp (n_dense_layers = max(0, num_layers-1) whenever the real
        first_k_dense_replace=3 >= num_layers), TWO layers: layer 0 dense +
        layer 1 MoE, not 2 dense layers as originally assumed.

    Those two results were never actually testing the same variable (they
    differ in both weight realism AND layer count/composition). This isolates
    the layer-count variable specifically: same random weights and the same
    dense+MoE composition (via ModelArgs' own n_dense_layers=1 default, so no
    loader/wrapper clamp logic needed), but built the same way as the already-
    passing single-Block test above -- plain ModifiedTransformer,
    use_mla_cache=False internal buffers, hand-written shard specs matching
    test_deepseek_attention_prefill_bh_galaxy /
    test_deepseek_v3_2_dense_block_bh_galaxy for attn/indexer/dense-ffn/norms,
    plus the MoE expert/router shard-spec convention from
    test_deepseek_v3_2_moe_block for layer 1 -- rather than reusing
    loader.load_shard_spec/DeepSeekV32ForCausalLM, which assume a different
    cache-wrapper setup and cost several iterations of plumbing mismatches
    (None cache buffers, a non-pytree dataclass output, missing activation
    shard specs) without ever reaching a real PCC number.

    If this reproduces the ~0.92 (vs. the single dense layer's >=0.99), that
    confirms cross-layer residual accumulation -- specifically the dense (0)
    -> MoE (1) handoff -- as the real source of the degradation. If it stays
    clean, the benchmark's ~0.92 must come from something this simplified
    forward-pass check doesn't exercise (e.g. its generation-loop/KV-cache
    machinery).

    No comparison_config is passed, so this uses the strict default
    (PCC >= 0.99) -- deliberately stricter than the ~0.92 the benchmark
    reported, so a failure's exception message reports the actual measured
    PCC for direct comparison against that ~0.92 figure.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Matches test_deepseek_v3_2_exp_tp_galaxy_2_layers exactly (batch_size=128
    # passed explicitly in test_llm_tp; input_sequence_length=128 printed in
    # its own benchmark config log) -- the last two variables separating this
    # test from the real benchmark after the causal-mask change above only
    # moved PCC from 0.9855 to 0.9824, still far above the benchmark's ~0.922.
    batch_size = 128
    seq_len = 128
    # n_dense_layers left at ModelArgs' own default (1): layer 0 dense,
    # layer 1 MoE -- the same composition the benchmark's clamp produces,
    # reached here without touching the loader's clamp/wrapper machinery.
    args = ModelArgs(
        n_layers=2, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )
    assert args.n_dense_layers == 1, "expected ModelArgs default n_dense_layers=1"

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    model.eval()

    layer0, layer1 = model.layers[0], model.layers[1]
    assert not isinstance(
        layer0.ffn, type(layer1.ffn)
    ), "expected layer 0 dense / layer 1 MoE -- got the same ffn type for both"

    mesh_shape = (4, 8)
    enable_sparse_mlp(layer1, mesh=mesh_shape, cluster_axis=0, config=args)

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Real causal mask (matches loader.py::DeepSeekV32ForCausalLM.forward's
    # self._causal_mask[:seqlen, :seqlen] construction exactly), not the
    # all-zero/fully-permissive mask every earlier test in this file used.
    # Broadcast to [batch, seq, seq] to match this test's mask shape
    # convention (the wrapper's own mask is unbatched, [seq, seq]).
    causal_mask_2d = torch.full(
        (seq_len, seq_len), float("-inf"), dtype=torch.bfloat16
    ).triu_(1)
    # .repeat (not .expand): mark_sharding needs a real, contiguous tensor,
    # not a stride-0 broadcast view.
    attention_mask = causal_mask_2d.unsqueeze(0).repeat(batch_size, 1, 1)
    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    stack = _TwoLayerStack(layer0, layer1)

    def get_shard_spec(stack, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}
        shard_specs[args[0]] = (batch_axis, None, None)  # hidden_states
        shard_specs[args[3]] = (batch_axis, None, None)  # attention_mask

        for layer in (stack.layer0, stack.layer1):
            attn = layer.attn
            shard_specs[attn.wq_a.weight] = (None, "model")
            shard_specs[attn.wkv_a.weight] = (None, "model")
            shard_specs[attn.wq_b.weight] = ("model", None)
            shard_specs[attn.wkv_b.weight] = ("model", None)
            shard_specs[attn.wo.weight] = (None, "model")
            shard_specs[attn.kv_cache] = (batch_axis, None, None)
            shard_specs[attn.pe_cache] = (batch_axis, None, None)

            idx = attn.indexer
            shard_specs[idx.wq_b.weight] = ("model", None)
            shard_specs[idx.wk.weight] = (None, "model")
            shard_specs[idx.weights_proj.weight] = (None, "model")
            shard_specs[idx.k_cache] = (batch_axis, None, None)

            shard_specs[layer.attn_norm.weight] = ("model",)
            shard_specs[layer.ffn_norm.weight] = ("model",)

        # layer0: dense MLP (loader.py::load_shard_spec convention).
        ffn0 = stack.layer0.ffn
        shard_specs[ffn0.w1.weight] = (batch_axis, "model")
        shard_specs[ffn0.w3.weight] = (batch_axis, "model")
        shard_specs[ffn0.w2.weight] = ("model", batch_axis)

        # layer1: sparse MoE (test_deepseek_v3_2_moe_block convention),
        # compound-sharded across both axes since enable_sparse_mlp stacked
        # the experts across the full (4,8)=32-device mesh.
        mlp = stack.layer1.ffn.mlp
        shard_specs[mlp.router.gate.weight] = (None, "model")
        shard_specs[mlp.experts.gate_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.up_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)
        shared = stack.layer1.ffn.shared_experts
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "model")
            shard_specs[shared.w3.weight] = (None, "model")
            shard_specs[shared.w2.weight] = ("model", None)

        return shard_specs

    # export_path/export_model_name are PJRT-level compile options, set via
    # torch_xla.set_custom_compile_options() -- CompilerConfig.to_torch_compile_options(),
    # not torch.compile()'s own `options` dict (that only reaches TTBackend's
    # dynamo-level config, which doesn't recognize export keys at all; a prior
    # attempt via torch_options silently produced an empty export directory).
    run_graph_test(
        stack,
        [hidden_states, 0, freqs_cis, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        compiler_config=CompilerConfig(
            export_path="/tmp/claude-1003/-home-ubuntu-hshah-tt-xla-dsa/9bcd6cb1-e2d7-407a-9395-1034fa048273/scratchpad/dsa_2layer_moe_ir",
            export_model_name="dsa_2l_moe_ir_check",
        ),
    )


def _make_page_table(users, blocks_per_user, device=None):
    return torch.arange(users * blocks_per_user, dtype=torch.int32, device=device).view(
        users, blocks_per_user
    )


def _pick_k_chunk_size(topk_tokens):
    # Mirrors integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py::TTIndexer._pick_k_chunk_size.
    for candidate in (128, 64, 32):
        if topk_tokens % candidate == 0:
            return candidate
    return 32


def _dsa_prefill_uses_sparse(seq_len, topk_tokens):
    # Mirrors integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py::dsa_prefill_uses_sparse.
    # Below topk_tokens, top-k over a causally-visible row keeps every visible
    # key, so sparse == dense exactly and the dense kernel is used instead.
    return seq_len >= topk_tokens


class TTOpsIndexer(nn.Module):
    """Prefill-only DSA indexer scoring via the real production custom ops
    (torch.ops.tt.indexer_score_dsa -> torch.ops.tt.topk_large_indices), as
    used by integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py::TTIndexer,
    instead of modified_model.py's own bf16_index bilinear score + Hadamard
    rotation.

    Reuses an existing Indexer's projection weights (wq_b/wk/k_norm/
    weights_proj) unchanged -- only the scoring/selection math changes.
    Deliberately drops the Hadamard rotation: that step exists only to smooth
    outliers ahead of FP8 quantization (see modified_model.py's
    rotate_activation), and the TT production indexer never quantizes -- it
    stays bf16 throughout -- so the rotation has no counterpart in
    indexer_score_dsa, whose documented math is a plain
    relu(q . k) * weights bilinear score.

    Publishes the selected indices on ``self.topk_indices`` (or ``None`` for
    dense attention), the same contract TTMLAAttentionBackendImpl reads via
    ``layer.indexer.topk_indices`` in production.
    """

    def __init__(self, orig_indexer, mesh, batch_size, max_seq_len, block_size=32):
        super().__init__()
        self.wq_b = orig_indexer.wq_b
        self.wk = orig_indexer.wk
        self.k_norm = orig_indexer.k_norm
        self.weights_proj = orig_indexer.weights_proj
        self.n_heads = orig_indexer.n_heads
        self.head_dim = orig_indexer.head_dim
        self.rope_head_dim = orig_indexer.rope_head_dim
        self.topk_tokens = orig_indexer.index_topk
        self.softmax_scale = orig_indexer.softmax_scale
        self.mesh = mesh
        self.k_chunk_size = _pick_k_chunk_size(self.topk_tokens)
        # Resolved once at construction, never inside forward: it calls into
        # torch_xla._XLAC, which dynamo cannot trace (mirrors TTIndexer.__init__).
        self._kernels_available = dsa_kernels_available()
        self.topk_indices = None

        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * blocks_per_user
        self.register_buffer(
            "k_cache",
            torch.zeros(num_blocks, 1, block_size, self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "page_table",
            _make_page_table(batch_size, blocks_per_user),
            persistent=False,
        )

    def _select(self, score, visible_count):
        indices = torch.ops.tt.topk_large_indices(score, self.topk_tokens)
        if self._kernels_available:
            return indices
        return topk_large_indices_mask_invalid_slots(indices, visible_count)

    def _prefill_seq_shard_plan(self, seq_len):
        # Mirrors TTIndexer._prefill_seq_shard_plan: on a multi-device mesh the
        # query sequence MUST be split across the model axis, or
        # indexer_score_dsa's per-device rank derivation aborts with "fullest
        # -device chunk window ... exceeds T" (a replicated query looks like
        # every device holds the full T-length window, which only satisfies
        # the op's bound when the mesh's model axis has size 1).
        axis_names = tuple(self.mesh.axis_names)
        try:
            model_axis = axis_names.index("model")
        except ValueError:
            model_axis = len(axis_names) - 1
        model_size = self.mesh.mesh_shape[model_axis]
        if model_size <= 1:
            return None, seq_len, model_axis
        align = 32 * model_size
        padded = ((seq_len + align - 1) // align) * align
        spec = tuple(axis_names[model_axis] if d == 2 else None for d in range(4))
        return spec, padded, model_axis

    def forward(self, x, qr, freqs_cis):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        # rope in indexer is not interleaved (matches Indexer.forward).
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        weights = self.weights_proj(x) * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale  # [b, s, n_heads, 1]

        q_op = q.transpose(1, 2).contiguous()  # [b, n_heads, s, head_dim]
        k_op = (
            k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2).contiguous()
        )  # [b, 1, s, head_dim]
        w_op = weights.transpose(1, 2).contiguous()  # [b, n_heads, s, 1]

        # Persist this chunk's indexer K (single-chunk prefill: chunk_start_idx=0).
        filled = self.k_cache
        for u in range(bsz):
            filled = torch.ops.tt.paged_fill_cache(
                filled,
                k_op[u : u + 1],
                self.page_table,
                batch_idx=torch.tensor([u], dtype=torch.int32, device=x.device),
            )
        self.k_cache.copy_(filled)

        if not _dsa_prefill_uses_sparse(seqlen, self.topk_tokens):
            self.topk_indices = None
            self.raw_scores = None
            return None, None

        seq_spec, padded_len, cluster_axis = self._prefill_seq_shard_plan(seqlen)
        key_op = k_op
        q_op_s, w_op_s = q_op, w_op
        if padded_len != seqlen:
            pad = padded_len - seqlen
            q_op_s = F.pad(q_op, (0, 0, 0, pad))
            w_op_s = F.pad(w_op, (0, 0, 0, pad))
            key_op = F.pad(k_op, (0, 0, 0, pad))

        visible_count = (
            torch.arange(padded_len, dtype=torch.int32, device=x.device) + 1
        ).view(1, 1, padded_len, 1)

        if seq_spec is not None:
            q_op_s = sharding_constraint_tensor(q_op_s, self.mesh, seq_spec)
            w_op_s = sharding_constraint_tensor(w_op_s, self.mesh, seq_spec)
            visible_count = sharding_constraint_tensor(
                visible_count, self.mesh, seq_spec
            )

        # Number of devices q's sequence dim was actually split into -- needed
        # by tt-mlir's decomposition fallback to reconstruct each device's
        # true causal window (see dsa_indexer.py's matching comment).
        num_devices = (
            self.mesh.mesh_shape[cluster_axis] if cluster_axis is not None else 1
        )

        # Both DSA ops require batch == 1, so score/select one user at a time.
        per_user_scores = []
        per_user_indices = []
        for u in range(bsz):
            score = torch.ops.tt.indexer_score_dsa(
                query=q_op_s[u : u + 1],
                key=key_op[u : u + 1],
                weights=w_op_s[u : u + 1],
                chunk_start_idx=0,
                cluster_axis=cluster_axis,
                num_devices=num_devices,
            )
            per_user_scores.append(score)
            per_user_indices.append(self._select(score, visible_count))
        scores = torch.cat(per_user_scores, dim=0)  # [b, 1, padded_len, padded_len]
        indices = torch.cat(per_user_indices, dim=0)  # [b, 1, padded_len, topk]

        if seq_spec is not None:
            scores = sharding_constraint_tensor(
                scores, self.mesh, (None, None, None, None)
            )
            indices = sharding_constraint_tensor(
                indices, self.mesh, (None, None, None, None)
            )

        # Row-slice both to drop the padded query rows; scores' last dim
        # (key positions) is left at padded_len -- indices reference that
        # same range, so gathering scores with indices downstream needs no
        # further adjustment.
        scores = scores[:, :, :seqlen, :]
        indices = indices[:, :, :seqlen, :]
        self.topk_indices = indices
        self.raw_scores = scores
        return indices, scores


class TTOpsMLA(nn.Module):
    """Prefill-only MLA that attends via the real production custom ops
    (torch.ops.tt.flash_mla_prefill / torch.ops.tt.sparse_sdpa), mirroring
    integrations/vllm_plugin/vllm_tt/attention_impls/attention_mla.py::
    TTMLAAttentionBackendImpl, instead of modified_model.py's own manual
    full nope/rope/v_head_dim einsum attention.

    Reuses an existing MLA's projection weights unchanged, and applies the
    same Q/K "absorption trick" production uses (W_UK_T/W_UV, derived from
    wkv_b's weight exactly as MLAAttention.process_weights_after_loading
    does) so K/V never leave the compressed kv_lora_rank latent space --
    the architectural difference that previously made an op-swap impossible.
    """

    def __init__(self, orig_mla, indexer, batch_size, max_seq_len, block_size=32):
        super().__init__()
        self.wq_a = orig_mla.wq_a
        self.q_norm = orig_mla.q_norm
        self.wq_b = orig_mla.wq_b
        self.wkv_a = orig_mla.wkv_a
        self.kv_norm = orig_mla.kv_norm
        self.wkv_b = orig_mla.wkv_b
        self.wo = orig_mla.wo
        self.n_local_heads = orig_mla.n_local_heads
        self.qk_nope_head_dim = orig_mla.qk_nope_head_dim
        self.qk_rope_head_dim = orig_mla.qk_rope_head_dim
        self.kv_lora_rank = orig_mla.kv_lora_rank
        self.v_head_dim = orig_mla.v_head_dim
        self.softmax_scale = orig_mla.softmax_scale
        self.indexer = indexer

        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * blocks_per_user
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                num_blocks,
                1,
                block_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "page_table",
            _make_page_table(batch_size, blocks_per_user),
            persistent=False,
        )

    def _absorbed_weights(self, device):
        # Mirrors MLAAttention.process_weights_after_loading exactly: wkv_b's
        # weight is [N*(P+V), L] (standard nn.Linear layout), so its transpose
        # is kv_b_proj_weight [L, N*(P+V)] there.
        #
        # Recomputed every call rather than cached: this test's harness runs
        # the SAME module instance once eagerly on CPU (for the golden output)
        # and once compiled on the XLA device, and W_UK_T/W_UV are plain
        # tensor attributes here (like production's own W_UK_T/W_UV -- see
        # the .to(device=...) below), not nn.Parameter/registered buffers, so
        # nothing moves them automatically when the model is moved to device.
        # A cached copy from the CPU pass would silently leak onto the device
        # pass and abort dynamo's fake-tensor device propagation.
        n = self.n_local_heads
        p = self.qk_nope_head_dim
        v = self.v_head_dim
        kv_b_proj_weight = self.wkv_b.weight.T.reshape(self.kv_lora_rank, n, p + v)
        W_UK, W_UV = kv_b_proj_weight.split([p, v], dim=-1)
        W_UV = W_UV.transpose(0, 1).contiguous().to(device=device)  # [N, L, V]
        W_UK_T = W_UK.permute(1, 2, 0).contiguous().to(device=device)  # [N, P, L]
        return W_UK_T, W_UV

    def forward(
        self, x, start_pos, freqs_cis, mask, past_key_value=None, cache_position=None
    ):
        # past_key_value/cache_position: Block.forward always passes these
        # (for the external-MLACache configuration); unused here since this
        # adapter is prefill-only and keeps its own internal paged caches.
        bsz, seqlen, _ = x.size()
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(
            bsz,
            seqlen,
            self.n_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # [b, s, 1, R]

        # Indexer runs first, from the same qr the absorption below reuses --
        # mirrors MultiHeadLatentAttentionWrapper.forward calling the indexer
        # before mla_attn. indexer is None when this instance is deliberately
        # DSA-free (index_n_heads=0), isolating pure dense MLA prefill/decode.
        if self.indexer is not None:
            self.indexer(x, qr, freqs_cis)
            topk_indices = self.indexer.topk_indices
        else:
            topk_indices = None

        act_dtype = q_pe.dtype
        W_UK_T, W_UV = self._absorbed_weights(device=x.device)
        q_nope_lat = torch.einsum("bsnp,npl->bsnl", q_nope, W_UK_T).to(act_dtype)
        q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [b, s, n, L+R]
        k_lat = torch.cat([kv_c.unsqueeze(2), k_pe], dim=-1)  # [b, s, 1, L+R]

        q_for_kernel = q_lat.transpose(1, 2).contiguous()  # [b, n, s, L+R]
        k_for_kernel = k_lat.transpose(1, 2).contiguous()  # [b, 1, s, L+R]

        if topk_indices is not None:
            out_lat = torch.cat(
                [
                    torch.ops.tt.sparse_sdpa(
                        query=q_for_kernel[u : u + 1],
                        kv=k_for_kernel[u : u + 1],
                        indices=topk_indices[u : u + 1],
                        v_dim=self.kv_lora_rank,
                        scale=self.softmax_scale,
                        k_chunk_size=self.indexer.k_chunk_size,
                    )
                    for u in range(bsz)
                ],
                dim=0,
            )  # [b, n, s, L]
        else:
            out_lat = torch.ops.tt.flash_mla_prefill(
                query=q_for_kernel,
                key=k_for_kernel,
                head_dim_v=self.kv_lora_rank,
                value=None,
                attn_mask=None,
                is_causal=True,
                scale=self.softmax_scale,
            )  # [b, n, s, L]

        out = torch.einsum("bnsl,nlv->bnsv", out_lat, W_UV).to(
            act_dtype
        )  # [b, n, s, V]
        out = out.transpose(1, 2).flatten(2)  # [b, s, n*V]

        # Persist tokens in the latent KV cache (single-chunk prefill).
        k_lat_for_fill = k_lat.transpose(1, 2)  # [b, 1, s, L+R]
        filled = self.kv_cache
        for u in range(bsz):
            filled = torch.ops.tt.paged_fill_cache(
                filled,
                k_lat_for_fill[u : u + 1],
                self.page_table,
                batch_idx=torch.tensor([u], dtype=torch.int32, device=x.device),
            )
        self.kv_cache.copy_(filled)

        return self.wo(out)

    def forward_decode(self, x, freqs_cis, cache_position):
        """One paged MLA decode step (S=1 new token per user), continuing from
        the SAME kv_cache/page_table a prior forward() prefill call filled --
        mirrors TTMLAAttentionBackendImpl._forward_decode. Unlike forward(),
        this is dense-only (no DSA/sparse_sdpa branch): the real E2E
        generation test that motivated this method never clears DSA's sparse
        predicate at its scale, so its decode path is always
        torch.ops.tt.paged_flash_mla_decode -- this method mirrors exactly
        that, not the sparse decode branch.

        Args:
            x: [b, 1, dim] hidden state for the single new token.
            freqs_cis: RoPE angles for this token's ABSOLUTE position (i.e.
                already indexed to cache_position, not position 0).
            cache_position: [b] int tensor, this token's absolute position
                (e.g. the prefill length, for the first decode step).
        """
        bsz, seqlen, _ = x.size()
        assert seqlen == 1, "forward_decode is single-new-token-per-user only"
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(
            bsz,
            seqlen,
            self.n_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # [b, 1, 1, R]

        act_dtype = q_pe.dtype
        W_UK_T, W_UV = self._absorbed_weights(device=x.device)
        q_nope_lat = torch.einsum("bsnp,npl->bsnl", q_nope, W_UK_T).to(act_dtype)
        q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [b, 1, n, L+R]
        k_lat = torch.cat([kv_c.unsqueeze(2), k_pe], dim=-1)  # [b, 1, 1, L+R]

        # Write this token's latent K into the paged cache at cache_position
        # (single call: paged_update_cache takes one absolute position per
        # user directly, unlike paged_fill_cache's per-user batch_idx loop).
        k_lat_for_update = k_lat.transpose(0, 1)  # [1, b, 1, L+R]
        updated_cache = torch.ops.tt.paged_update_cache(
            self.kv_cache, k_lat_for_update, cache_position, self.page_table
        )
        self.kv_cache.copy_(updated_cache)

        out_lat = torch.ops.tt.paged_flash_mla_decode(
            query=q_lat.transpose(0, 1),  # [1, b, n, L+R]
            key=self.kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table=self.page_table,
            value=None,
            is_causal=True,
            attn_mask=None,
            cur_pos_tensor=cache_position,
            scale=self.softmax_scale,
        )  # [1, b, n, L]

        out_lat = out_lat.reshape(bsz, self.n_local_heads, self.kv_lora_rank)
        out = torch.einsum("bnl,nlv->bnv", out_lat, W_UV).to(act_dtype)  # [b, n, V]
        out = out.reshape(bsz, self.n_local_heads * self.v_head_dim)
        return self.wo(out)


class _IndexerScoreAndTopK(nn.Module):
    """Test-only wrapper exposing TTOpsIndexer's raw (indices, score) pair,
    for a comparator that checks scoring and selection separately from a
    whole-model-output PCC (see _dsa_indexer_score_topk_comparator)."""

    def __init__(self, ttops_indexer):
        super().__init__()
        self.indexer = ttops_indexer

    def forward(self, x, qr, freqs_cis):
        return self.indexer(x, qr, freqs_cis)


def _dsa_indexer_score_topk_comparator(device_output, golden_output, args, kwargs):
    """Comparator for TTOpsIndexer's (indices, score) sparse-path output.

    A whole-model PCC conflates two different questions: did
    indexer_score_dsa's scoring diverge, and did topk_large_indices' hard,
    discontinuous top-k selection diverge in a way that actually changes
    which keys get attended to. torch.topk (and tt.topk_large_indices)
    ordering is not required to be identical between golden and device --
    ties near the k-th boundary can legitimately select a different key with
    a near-equal score -- so comparing indices directly is too strict, the
    same reasoning tests/torch/ops/test_topk.py's topk_both_comparator /
    _topk_vllm_comparator apply. This mirrors that pattern, split into the
    two checks the user asked for:

      1) indexer_score_dsa's raw score tensor: a normal PCC, since both
         sides compute the identical causal-masked relu(q.k)*weights
         bilinear score and should match closely regardless of what
         top-k later selects from it.
      2) topk_large_indices' selection: gather EACH side's own score using
         its OWN indices (not a shared golden index set), then compare the
         two gathered results via cosine similarity/PCC. Because
         topk_large_indices returns indices sorted by descending score,
         position i in the gathered result is "the i-th highest score in
         this row" on each side -- so this checks "did selection surface
         similarly-ranked keys," not "did it choose the literal same index."
    """
    device_indices, device_score = device_output
    golden_indices, golden_score = golden_output

    device_score = device_score.cpu()
    device_indices = device_indices.cpu()

    assert golden_score is not None and device_score is not None, (
        "indexer returned no score -- seq_len must be >= index_topk for the "
        "sparse path (dsa_prefill_uses_sparse) to run at all"
    )

    # 1) Scores should match closely.
    pcc = compute_pcc(golden_score, device_score)
    print(f"\n  indexer_score_dsa PCC: {pcc}")
    assert pcc > 0.99, f"indexer_score_dsa PCC: {pcc} (required > 0.99)"

    # 2) Gather each side's own score with its own indices. Rows near the
    # causal boundary have fewer than index_topk visible keys, so their
    # tail slots are invalid. Two different invalidity signals are checked,
    # because which one fires depends on whether topk_large_indices' fused
    # TTNN kernel was actually promoted for this compile:
    #   - kernel-promoted: the op implements the -inf -> sentinel contract
    #     itself (-1 once cast from its native uint32 0xFFFFFFFF, or a
    #     genuinely huge value if some other cast path is hit) -- caught by
    #     the index-range check.
    #   - NOT promoted: TTNNResolveComposites inlines a plain ttir.topk,
    #     which does *not* honor that contract for -inf ties -- it returns an
    #     ordinary, in-range index instead of the sentinel (see
    #     topk_large_indices_mask_invalid_slots' docstring: "the decomposition
    #     returned keys 38/36/39/... alongside key 0"). An index-range check
    #     alone would miss this, so any gathered value that comes back
    #     non-finite (the causal mask's real -inf) is *also* excluded --
    #     valid regardless of which index reached it.
    key_len = golden_score.shape[-1]

    def gather_valid(score, indices):
        # int64 first: torch has no comparison kernels for uint32 (the
        # topk_large_indices kernel's native output dtype), and a value-
        # preserving cast (not a bit-reinterpret) correctly turns its
        # 0xFFFFFFFF sentinel into a large positive int64, still caught by
        # `>= key_len` below; an int32 -1 sentinel sign-extends to -1.
        indices = indices.to(torch.int64)
        index_invalid = (indices < 0) | (indices >= key_len)
        gather_idx = indices.clamp(min=0, max=key_len - 1)
        gathered = torch.gather(score, -1, gather_idx)
        score_invalid = ~torch.isfinite(gathered.float())
        invalid = index_invalid | score_invalid
        return gathered[~invalid]

    golden_gathered = gather_valid(golden_score, golden_indices)
    device_gathered = gather_valid(device_score, device_indices)

    # Diagnostic: per-query-row valid count (summed over batch), to see
    # whether a mismatch correlates with the model-axis shard boundary
    # (every 32 rows in the padded 256-wide query range this indexer pads
    # seq_len=128 up to, per _prefill_seq_shard_plan with model_size=8).
    def per_row_valid_count(score, indices):
        indices64 = indices.to(torch.int64)
        index_invalid = (indices64 < 0) | (indices64 >= key_len)
        gather_idx = indices64.clamp(min=0, max=key_len - 1)
        gathered = torch.gather(score, -1, gather_idx)
        valid = torch.isfinite(gathered.float()) & ~index_invalid
        return valid.sum(dim=(0, 1, 3))  # [seqlen] summed over batch/head

    golden_per_row = per_row_valid_count(golden_score, golden_indices)
    device_per_row = per_row_valid_count(device_score, device_indices)
    print(f"\n  golden valid/row (32-row chunks): {golden_per_row[::32].tolist()}")
    print(f"  device valid/row (32-row chunks): {device_per_row[::32].tolist()}")
    mismatched_rows = (golden_per_row != device_per_row).nonzero().flatten()
    print(
        f"  rows with a count mismatch: {mismatched_rows.numel()}/{golden_per_row.numel()}"
        f" (first few: {mismatched_rows[:10].tolist()})"
    )

    assert golden_gathered.numel() == device_gathered.numel(), (
        f"gathered element count mismatch: golden={golden_gathered.numel()} "
        f"device={device_gathered.numel()} -- indexer_score_dsa's causal "
        "mask makes each row's valid-key count a pure function of row "
        "position, so it should be identical on both sides"
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0).float(),
        golden_gathered.flatten().unsqueeze(0).float(),
    )
    assert (
        cos_sim > 0.99
    ), f"gathered indexer score cosine similarity: {cos_sim.item()} (required > 0.99)"


@pytest.mark.bh_galaxy
def test_deepseek_indexer_dsa_topk_bh_galaxy():
    """Isolated DSA sparse-selection check for TTOpsIndexer, using
    _dsa_indexer_score_topk_comparator instead of a whole-output PCC.

    Same real ops as the sparse branch of
    test_deepseek_v3_2_dense_moe_2layer_transformer_ttops_bh_galaxy
    (torch.ops.tt.indexer_score_dsa -> torch.ops.tt.topk_large_indices), same
    32-device Galaxy mesh and batch/seq/index_topk config -- isolated to just
    the indexer so a scoring/selection mismatch isn't conflated with whatever
    the rest of the model does downstream.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 128
    seq_len = 128
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_topk=64,
    )
    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    model.eval()

    mesh_shape = (4, 8)
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    orig_indexer = model.layers[0].attn.indexer
    ttops_indexer = TTOpsIndexer(
        orig_indexer, mesh, batch_size=batch_size, max_seq_len=seq_len
    )
    wrapper = _IndexerScoreAndTopK(ttops_indexer)

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    qr = torch.randn((batch_size, seq_len, args.q_lora_rank), dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    def get_shard_spec(wrapper, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        idx = wrapper.indexer
        shard_specs = {
            args[0]: (batch_axis, None, None),  # hidden_states
            args[1]: (batch_axis, None, None),  # qr
            idx.wq_b.weight: ("model", None),
            idx.wk.weight: (None, "model"),
            idx.weights_proj.weight: (None, "model"),
            idx.k_cache: (None, None, None, None),
            idx.page_table: (None, None),
        }
        return shard_specs

    run_graph_test(
        wrapper,
        [hidden_states, qr, freqs_cis],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        custom_comparator=_dsa_indexer_score_topk_comparator,
    )


@pytest.mark.bh_galaxy
def test_deepseek_v3_2_dense_moe_2layer_transformer_ttops_bh_galaxy():
    """Same 2-layer (dense + MoE) PCC check as
    test_deepseek_v3_2_dense_moe_2layer_transformer_bh_galaxy, but with both
    layers' MLA attention and DSA indexer replaced by TTOpsMLA/TTOpsIndexer --
    thin adapters that reuse modified_model.py's own projection weights but
    attend/score via the SAME torch.ops.tt.* custom ops
    (flash_mla_prefill/sparse_sdpa, indexer_score_dsa/topk_large_indices,
    paged_fill_cache) that integrations/vllm_plugin/vllm_tt/ uses in the real
    E2E path, instead of modified_model.py's own manual einsum attention and
    Hadamard-rotated bf16_index score.

    Verifying the MoE point (see the IR-export check on the sibling test
    above) found the MoE path already faithful; the indexer and MLA attention
    were not -- this closes that gap so a PCC regression here is directly
    attributable to the real production op stack, not an architectural
    stand-in.

    index_topk is set to 64 (legal: in [16, 2048], multiple of 32) rather
    than ModelArgs' production default of 2048, specifically so this test's
    seq_len=128 exercises the SPARSE indexer_score_dsa/topk_large_indices/
    sparse_sdpa path (dsa_prefill_uses_sparse requires seq_len >= index_topk)
    instead of silently falling back to dense flash_mla_prefill the whole
    time, which is what index_topk=2048 would still correctly do for this
    scale but would leave the sparse path unexercised.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 128
    seq_len = 128
    args = ModelArgs(
        n_layers=2,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_topk=64,
    )
    assert args.n_dense_layers == 1, "expected ModelArgs default n_dense_layers=1"

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    model.eval()

    layer0, layer1 = model.layers[0], model.layers[1]
    assert not isinstance(
        layer0.ffn, type(layer1.ffn)
    ), "expected layer 0 dense / layer 1 MoE -- got the same ffn type for both"

    mesh_shape = (4, 8)
    enable_sparse_mlp(layer1, mesh=mesh_shape, cluster_axis=0, config=args)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    for layer in (layer0, layer1):
        orig_indexer = layer.attn.indexer
        ttops_indexer = TTOpsIndexer(
            orig_indexer, mesh, batch_size=batch_size, max_seq_len=seq_len
        )
        layer.attn = TTOpsMLA(
            layer.attn,
            ttops_indexer,
            batch_size=batch_size,
            max_seq_len=seq_len,
        )

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    causal_mask_2d = torch.full(
        (seq_len, seq_len), float("-inf"), dtype=torch.bfloat16
    ).triu_(1)
    attention_mask = causal_mask_2d.unsqueeze(0).repeat(batch_size, 1, 1)
    freqs_cis = model.freqs_cis[0:seq_len]

    stack = _TwoLayerStack(layer0, layer1)

    def get_shard_spec(stack, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}
        shard_specs[args[0]] = (batch_axis, None, None)  # hidden_states
        shard_specs[args[3]] = (
            batch_axis,
            None,
            None,
        )  # attention_mask (unused by TTOpsMLA)

        for layer in (stack.layer0, stack.layer1):
            attn = layer.attn
            shard_specs[attn.wq_a.weight] = (None, "model")
            shard_specs[attn.wkv_a.weight] = (None, "model")
            shard_specs[attn.wq_b.weight] = ("model", None)
            shard_specs[attn.wkv_b.weight] = ("model", None)
            shard_specs[attn.wo.weight] = (None, "model")
            # kv_cache/k_cache are now paged 4D tensors [num_blocks, 1,
            # block_size, head_dim], written by a per-user paged_fill_cache
            # loop (one call per row of the fill-value tensor). Batch-sharding
            # the cache/page_table on dim 0 (matching the sibling test's 3D
            # buffers) made Shardy's propagation fail across that loop
            # ("Could not apply propagated tensor shardings" at a `slice` op)
            # -- confirmed by bisection: disabling the fill loop alone made
            # the same dense-path graph compile and run cleanly. Leaving the
            # cache/page_table fully replicated sidesteps the propagation
            # conflict; each per-user call still only writes its own slice,
            # so this changes no output, only how the buffer is sharded.
            shard_specs[attn.kv_cache] = (None, None, None, None)
            shard_specs[attn.page_table] = (None, None)

            idx = attn.indexer
            shard_specs[idx.wq_b.weight] = ("model", None)
            shard_specs[idx.wk.weight] = (None, "model")
            shard_specs[idx.weights_proj.weight] = (None, "model")
            shard_specs[idx.k_cache] = (None, None, None, None)
            shard_specs[idx.page_table] = (None, None)

            shard_specs[layer.attn_norm.weight] = ("model",)
            shard_specs[layer.ffn_norm.weight] = ("model",)

        ffn0 = stack.layer0.ffn
        shard_specs[ffn0.w1.weight] = (batch_axis, "model")
        shard_specs[ffn0.w3.weight] = (batch_axis, "model")
        shard_specs[ffn0.w2.weight] = ("model", batch_axis)

        mlp = stack.layer1.ffn.mlp
        shard_specs[mlp.router.gate.weight] = (None, "model")
        shard_specs[mlp.experts.gate_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.up_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)
        shared = stack.layer1.ffn.shared_experts
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "model")
            shard_specs[shared.w3.weight] = (None, "model")
            shard_specs[shared.w2.weight] = ("model", None)

        return shard_specs

    run_graph_test(
        stack,
        [hidden_states, 0, freqs_cis, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        compiler_config=CompilerConfig(
            export_path="/tmp/claude-1003/-home-ubuntu-hshah-tt-xla-dsa/9bcd6cb1-e2d7-407a-9395-1034fa048273/scratchpad/dsa_2layer_ttops_ir",
            export_model_name="dsa_2l_ttops_ir_check",
        ),
    )


class _MLAPrefillThenDecode(nn.Module):
    """Runs a real prefill through TTOpsMLA.forward (filling its paged
    kv_cache/page_table for real), then ONE decode step through
    TTOpsMLA.forward_decode continuing from that same cache -- as a single
    traced forward, so the CPU-golden and device passes each maintain their
    OWN internally-consistent cache state throughout (two separate
    run_graph_test calls would let the device pass's cache state leak into
    what should be a CPU-only golden for the decode step, since kv_cache is a
    mutated buffer shared by both backends' otherwise-independent runs).
    Returns only the decode step's output -- prefill's output is discarded,
    since decode correctness is what's under test here.
    """

    def __init__(self, mla):
        super().__init__()
        self.mla = mla

    def forward(
        self,
        hidden_states,
        freqs_cis_prefill,
        next_token,
        freqs_cis_decode,
        cache_position,
    ):
        self.mla(hidden_states, 0, freqs_cis_prefill, None)
        return self.mla.forward_decode(next_token, freqs_cis_decode, cache_position)


@pytest.mark.bh_galaxy
def test_deepseek_v3_2_mla_prefill_decode_bh_galaxy():
    """Prefill (128 tokens) then ONE dense MLA decode step, on the real
    mesh_shape=[8,4] / batch=1 regime test_tensor_parallel_generation_deepseek_v32_full
    (the full 61-layer E2E generation test) actually runs at.

    Two things distinguish this from every other test in this file:

    1. batch_size=1 < the mesh's batch axis size (8), so hidden states are
       REPLICATED across the batch axis rather than sharded -- every other
       test here uses batch>=4, which stays on the batch-SHARDED branch of
       the same ternary. This is a genuinely different, previously-untested
       sharding regime, and the real E2E test's max_num_seqs=1 means it is
       the ONLY regime that test actually runs in.
    2. index_n_heads=0 disables the indexer/DSA entirely, matching the real
       E2E test's own dense-only regime at its scale (see that test's
       docstring: the sparse predicate never clears at model_len=128 /
       index_topk=2048) -- so only torch.ops.tt.flash_mla_prefill (prefill)
       and torch.ops.tt.paged_update_cache + paged_flash_mla_decode (decode)
       are exercised, the exact two ops real E2E generation actually calls
       for attention at every step.

    Isolating these two factors (previously only tested together, at full
    model scale, inside the incoherent E2E run) tells us whether the dense
    MLA decode step itself -- never before tested against a CPU golden on
    real multi-device hardware -- is a source of the E2E run's incoherence,
    independent of MoE, RMSNorm/RoPE precision compounding over 61 layers, or
    anything downstream of attention.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 1
    seq_len = 128
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_n_heads=0,
    )
    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    _fix_layernorm_dtype(model)
    model.eval()

    orig_mla = model.layers[0].attn
    assert orig_mla.indexer is None, "expected DSA disabled via index_n_heads=0"
    mla = TTOpsMLA(
        orig_mla, indexer=None, batch_size=batch_size, max_seq_len=args.max_seq_len
    )
    wrapper = _MLAPrefillThenDecode(mla)

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    next_token = torch.randn((batch_size, 1, args.dim), dtype=torch.bfloat16)
    freqs_cis_prefill = model.freqs_cis[0:seq_len]
    freqs_cis_decode = model.freqs_cis[seq_len : seq_len + 1]
    cache_position = torch.full((batch_size,), seq_len, dtype=torch.int32)

    mesh_shape = (8, 4)
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(wrapper, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        mla = wrapper.mla
        return {
            args[0]: (batch_axis, None, None),  # hidden_states
            args[2]: (batch_axis, None, None),  # next_token
            args[4]: (batch_axis,),  # cache_position
            mla.wq_a.weight: (None, "model"),
            mla.wkv_a.weight: (None, "model"),
            mla.wq_b.weight: ("model", None),
            mla.wkv_b.weight: ("model", None),
            mla.wo.weight: (None, "model"),
            mla.kv_cache: (None, None, None, None),
            mla.page_table: (None, None),
        }

    run_graph_test(
        wrapper,
        [
            hidden_states,
            freqs_cis_prefill,
            next_token,
            freqs_cis_decode,
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

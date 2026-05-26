# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from safetensors.torch import load_file as safetensors_load_file
from torch_xla.distributed.spmd import Mesh
from transformers import PreTrainedTokenizerFast
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

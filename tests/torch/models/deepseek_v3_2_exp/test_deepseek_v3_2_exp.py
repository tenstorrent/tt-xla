# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import re

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.testers.compiler_config import CompilerConfig
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from safetensors import safe_open
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation

# This model is modified from the original deepseek_v3_2_exp model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
#    - fast_hadamard_transform requires a CUDA enviroment and fails to install
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
#    - the original implementation (kernel.py) relies on custom tilelang kernels not supported on TT
# 3. Avoid torch.view_as_complex/view_as_real operations

DEEPSEEK_V3_REPO = "deepseek-ai/DeepSeek-V3.2"


def _rename_hf_key(ckpt_key, n_dense_layers=1):
    """Rename a HuggingFace checkpoint key to match modified_model.py state dict naming."""
    key = ckpt_key

    # Strip "model." prefix
    if key.startswith("model."):
        key = key[len("model.") :]

    # Skip FP8 quantization scale keys
    if "weight_scale_inv" in key:
        return None

    # Top-level renames
    key = key.replace("lm_head.", "head.")
    key = key.replace("embed_tokens.", "embed.")

    # Layer norms
    key = re.sub(r"(layers\.\d+\.)input_layernorm\.", r"\1attn_norm.", key)
    key = re.sub(r"(layers\.\d+\.)post_attention_layernorm\.", r"\1ffn_norm.", key)

    # Attention (indexer must come before other self_attn renames)
    key = key.replace("self_attn.indexer.", "attn.indexer.")
    key = key.replace("self_attn.q_a_proj.", "attn.wq_a.")
    key = key.replace("self_attn.q_b_proj.", "attn.wq_b.")
    key = key.replace("self_attn.q_a_layernorm.", "attn.q_norm.")
    key = key.replace("self_attn.kv_a_proj_with_mqa.", "attn.wkv_a.")
    key = key.replace("self_attn.kv_b_proj.", "attn.wkv_b.")
    key = key.replace("self_attn.kv_a_layernorm.", "attn.kv_norm.")
    key = key.replace("self_attn.o_proj.", "attn.wo.")

    # MoE routed experts (before bare mlp renames)
    key = re.sub(r"mlp\.experts\.(\d+)\.gate_proj\.", r"ffn.experts.\1.w1.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.down_proj\.", r"ffn.experts.\1.w2.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.up_proj\.", r"ffn.experts.\1.w3.", key)

    # MoE shared experts (explicit shared_experts prefix)
    key = key.replace("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1.")
    key = key.replace("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2.")
    key = key.replace("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3.")

    # MoE router gate
    key = key.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
    key = key.replace("mlp.gate.", "ffn.gate.")

    # Bare mlp.{gate_proj,down_proj,up_proj}: only valid for dense layers.
    # For MoE layers, these have incompatible shapes — skip them.
    # (MoE shared experts are loaded via explicit mlp.shared_experts.* keys above.)
    layer_m = re.match(r"layers\.(\d+)\.", key)
    if layer_m:
        layer_id = int(layer_m.group(1))
        if layer_id < n_dense_layers:
            key = key.replace("mlp.gate_proj.", "ffn.w1.")
            key = key.replace("mlp.down_proj.", "ffn.w2.")
            key = key.replace("mlp.up_proj.", "ffn.w3.")
        elif (
            "mlp.gate_proj." in key or "mlp.down_proj." in key or "mlp.up_proj." in key
        ):
            return None  # Skip — incompatible shape for MoE shared experts

    return key


def load_deepseek_config(repo_id=DEEPSEEK_V3_REPO):
    """Download and parse the HuggingFace config.json into ModelArgs fields."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)

    # Map HuggingFace config keys to ModelArgs field names
    return ModelArgs(
        vocab_size=hf_cfg["vocab_size"],
        dim=hf_cfg["hidden_size"],
        inter_dim=hf_cfg["intermediate_size"],
        moe_inter_dim=hf_cfg["moe_intermediate_size"],
        n_layers=hf_cfg["num_hidden_layers"],
        n_dense_layers=hf_cfg.get("first_k_dense_replace", 1),
        n_heads=hf_cfg["num_attention_heads"],
        n_routed_experts=hf_cfg.get("n_routed_experts", 256),
        n_shared_experts=hf_cfg.get("n_shared_experts", 1),
        n_activated_experts=hf_cfg.get("num_experts_per_tok", 8),
        n_expert_groups=hf_cfg.get("n_group", 8),
        n_limited_groups=hf_cfg.get("topk_group", 4),
        score_func=hf_cfg.get("scoring_func", "sigmoid"),
        route_scale=hf_cfg.get("routed_scaling_factor", 2.5),
        q_lora_rank=hf_cfg.get("q_lora_rank", 1536),
        kv_lora_rank=hf_cfg.get("kv_lora_rank", 512),
        qk_nope_head_dim=hf_cfg.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=hf_cfg.get("qk_rope_head_dim", 64),
        v_head_dim=hf_cfg.get("v_head_dim", 128),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        index_n_heads=hf_cfg.get("index_n_heads", 0),
        index_head_dim=hf_cfg.get("index_head_dim", 128),
        index_topk=hf_cfg.get("index_topk", 2048),
    )


def load_deepseek_weights(
    model, repo_id=DEEPSEEK_V3_REPO, n_layers=2, n_dense_layers=1
):
    """Load pretrained weights from a HuggingFace repo into the model.

    The HF checkpoint uses different key naming than modified_model.py,
    so keys are remapped during loading.  Only the safetensors shards
    containing the first ``n_layers`` layers (plus top-level weights
    like embed, norm, head) are downloaded.

    Weights not found in the checkpoint (e.g. Indexer parameters, caches)
    remain at their initialized values.
    """
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Build ckpt_key -> (model_key, shard_file) mapping, filtering by layer
    needed_shards = set()
    needed_keys = {}  # ckpt_key -> model_key
    for ckpt_key, shard_file in weight_map.items():
        model_key = _rename_hf_key(ckpt_key, n_dense_layers)
        if model_key is None:
            continue
        # Filter to only needed layers
        layer_m = re.match(r"layers\.(\d+)\.", model_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue
        needed_shards.add(shard_file)
        needed_keys[ckpt_key] = model_key

    # Download needed shards and build state dict
    state_dict = {}
    for shard_name in sorted(needed_shards):
        print(f"[weights] loading shard: {shard_name}")
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed_keys:
                    state_dict[needed_keys[key]] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[weights] loaded {len(state_dict)} tensors from {repo_id}. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    if unexpected:
        print(
            f"[weights] first 20 unexpected keys (checkpoint): {sorted(unexpected)[:20]}"
        )
    if missing:
        print(f"[weights] first 20 missing keys (model): {sorted(missing)[:20]}")

    # Verify all weight parameters were loaded (non-weight buffers like caches may remain at init)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    not_loaded = model_keys - loaded_keys
    print(f"[weights] model keys not loaded: {sorted(not_loaded)}")

    return model


# def test_deepseek_modified_transformer_single_layer():
#     xr.set_device_type("TT")

#     # Create model args with a single layer for testing
#     args = ModelArgs(
#         n_layers=1,
#     )

#     model = ModifiedTransformer(args)

#     model = model.to(torch.bfloat16)

#     model = model.eval()
#     compiled_model = torch.compile(model, backend="tt")

#     batch_size = 1
#     seq_len = 32
#     tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

#     device = torch_xla.device()
#     tokens = tokens.to(device)
#     compiled_model = compiled_model.to(device)

#     with torch.no_grad():
#         output = compiled_model(tokens)
#         output.to("cpu")


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


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_attention_prefill(batch_size):
    xr.set_device_type("TT")
    seq_len = 32
    args = ModelArgs(n_layers=1, max_batch_size=batch_size, max_seq_len=seq_len * 2)

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Prefill branch expects mask shape (bsz, seqlen, seqlen) for index_mask += mask
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)

    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
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


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_indexer(batch_size):
    xr.set_device_type("TT")

    seq_len = 32
    args = ModelArgs(n_layers=1, max_batch_size=batch_size, max_seq_len=seq_len * 2)

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
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
    mesh_shape = (4, 8)
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


@pytest.mark.llmbox
def test_deepseek_v3_2_moe_only():
    """Test MoE MLP only (no attention) with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 64
    seq_len = 128
    args = ModelArgs(
        n_layers=2,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    # Replace MoE module in block layer 1, then extract the replaced ffn
    block = model.layers[1]
    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    moe = block.ffn  # Now A2aSparseMLPWithSharedExperts
    moe.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(moe, args, kwargs):
        shard_specs = {}

        # x: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        mlp = moe.mlp if hasattr(moe, "mlp") else moe
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # Shared experts
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        moe,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("seq_len", [1, 32, 128])
def test_deepseek_v3_2_layer_sparse_moe(batch_size, seq_len):
    """Test single MoE Block with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    args = ModelArgs(
        n_layers=2,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    # Create full model to get freqs_cis, then extract MoE block (layer 1)
    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    block = model.layers[1]  # layer_id=1 >= n_dense_layers=1 → MoE
    freqs_cis = model.freqs_cis[:seq_len]

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    block.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(block, args, kwargs):
        shard_specs = {}

        # x: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        # Attention weights — all parallelism on _axis_0 (matches hidden on _axis_0)
        attn = block.attn
        shard_specs[attn.wq_b.weight] = ("_axis_0", None)
        shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
        shard_specs[attn.wo.weight] = (None, "_axis_0")
        shard_specs[attn.wq_a.weight] = (None, "_axis_0")
        shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

        # KV caches [max_batch, max_seq, dim] — batch on _axis_1
        shard_specs[attn.kv_cache] = ("_axis_1", None, None)
        shard_specs[attn.pe_cache] = ("_axis_1", None, None)

        # Indexer
        if attn.indexer is not None:
            shard_specs[attn.indexer.wq_b.weight] = ("_axis_0", None)
            shard_specs[attn.indexer.wk.weight] = (None, "_axis_0")
            shard_specs[attn.indexer.weights_proj.weight] = (None, "_axis_0")
            shard_specs[attn.indexer.k_cache] = ("_axis_1", None, None)

        # A2aSparseMLP
        ffn = block.ffn
        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_proj_bias] = (
            ("_axis_0", "_axis_1"),
            None,
        )
        shard_specs[mlp.experts.up_proj_bias] = (
            ("_axis_0", "_axis_1"),
            None,
        )
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # Shared experts
        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        # Norms
        shard_specs[block.attn_norm.weight] = ("_axis_0",)
        shard_specs[block.ffn_norm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        block,
        [hidden_states, None, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
def test_deepseek_v3_2_full_sparse_moe():
    """Test full DeepseekV3-2 Transformer with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    token_ids = [
        671,
        6102,
        294,
        8760,
        344,
        11111,
        14,
        260,
        5217,
        6354,
        362,
        2783,
        14,
        13556,
        14,
        17224,
        87191,
        305,
    ]

    batch_size = 32
    seq_len = len(token_ids)

    args = load_deepseek_config()
    # args.n_dense_layers = 1
    args.n_layers = 1
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2
    print(f"[config] {args}")

    model = ModifiedTransformer(args)
    load_deepseek_weights(
        model, n_layers=args.n_layers, n_dense_layers=args.n_dense_layers
    )
    model = model.to(torch.bfloat16)
    # head is intentionally float32 in the original model (logits computed in fp32),
    # but model.to(bf16) converts it. Restore to float32 to match forward's .float() call.
    model.head = model.head.to(torch.float32)

    mesh_shape = (4, 8)
    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=args,
    )

    model.eval()

    single_sequence = torch.tensor(token_ids).long()
    tokens = single_sequence.unsqueeze(0).expand(batch_size, seq_len)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}

        # Input tokens [batch, seq]
        shard_specs[args[0]] = ("_axis_1", None)

        # Embedding
        shard_specs[model.embed.weight] = (None, "_axis_0")

        # Per-layer sharding
        for layer in model.layers:
            attn = layer.attn

            # MLA attention weights — all parallelism on _axis_0
            shard_specs[attn.wq_b.weight] = ("_axis_0", None)
            shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
            shard_specs[attn.wo.weight] = (None, "_axis_0")
            shard_specs[attn.wq_a.weight] = (None, "_axis_0")
            shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

            # KV caches [max_batch, max_seq, dim] — batch on _axis_1
            shard_specs[attn.kv_cache] = ("_axis_1", None, None)
            shard_specs[attn.pe_cache] = ("_axis_1", None, None)

            # Indexer
            if attn.indexer is not None:
                shard_specs[attn.indexer.wq_b.weight] = ("_axis_0", None)
                shard_specs[attn.indexer.wk.weight] = (None, "_axis_0")
                shard_specs[attn.indexer.weights_proj.weight] = (None, "_axis_0")
                shard_specs[attn.indexer.k_cache] = ("_axis_1", None, None)

            # FFN sharding (MoE or dense)
            ffn = layer.ffn
            if hasattr(ffn, "mlp"):
                # A2aSparseMLPWithSharedExperts (MoE layer)
                mlp = ffn.mlp
                shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
                shard_specs[mlp.experts.gate_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.up_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.gate_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )
                shard_specs[mlp.experts.up_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )
                shard_specs[mlp.experts.down_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )

                # Shared experts (MLP with w1/w2/w3)
                shared = getattr(ffn, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.w1.weight] = (None, "_axis_0")
                    shard_specs[shared.w3.weight] = (None, "_axis_0")
                    shard_specs[shared.w2.weight] = ("_axis_0", None)
            else:
                # Dense MLP
                shard_specs[ffn.w1.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w3.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w2.weight] = ("_axis_0", "_axis_1")

            # Norms
            shard_specs[layer.attn_norm.weight] = ("_axis_0",)
            shard_specs[layer.ffn_norm.weight] = ("_axis_0",)

        # Final norm and head
        shard_specs[model.norm.weight] = ("_axis_0",)
        shard_specs[model.head.weight] = (None, "_axis_0")

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        model,
        [tokens],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )

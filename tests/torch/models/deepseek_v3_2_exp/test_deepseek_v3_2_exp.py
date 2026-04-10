# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sharding import sharding_constraint_hook

from tests.utils import failed_ttmlir_compilation

# This model is modified from the original deepseek_v3_2_exp model.py to:
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


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_deepseek_attention_prefill(batch_size, seq_len):
    xr.set_device_type("TT")
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_topk=16,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Prefill branch expects mask shape (bsz, seqlen, seqlen) for index_mask += mask
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)

    # Create a (batch_size, seq_len, index_topk) tensor of valid indices.
    # Each entry along the last axis contains values from 0 to seq_len-1, in random order per batch/position.
    topk_indices = torch.stack(
        [
            torch.stack(
                [torch.randperm(seq_len)[: args.index_topk] for _ in range(seq_len)]
            ).unsqueeze(1)
            for _ in range(batch_size)
        ]
    ).squeeze(
        2
    )  # shape: (batch_size, seq_len, index_topk)

    attention.prepopulated_topk_indices = topk_indices

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
        pcc=PccConfig(enabled=True, required_pcc=0.99),
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
@pytest.mark.parametrize("prefill_seq_len", [32, 128, 512, 2048])
def test_deepseek_attention_decode(batch_size, prefill_seq_len, request):
    _XFAIL_CONFIGS = {
        (128, 32),
        (128, 64),
        (512, 32),
        (512, 64),
        (2048, 4),
        (2048, 32),
        (2048, 64),
    }
    if (prefill_seq_len, batch_size) in _XFAIL_CONFIGS:
        request.applymarker(
            pytest.mark.xfail(
                reason="Low PCC due to ttir.gather lowering bug - https://github.com/tenstorrent/tt-xla/issues/3726"
            )
        )

    xr.set_device_type("TT")

    # Decode-specific parameters
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        # index_topk=prefill_seq_len // 2,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    # Create decode input: single token only
    hidden_states = torch.randn(
        (batch_size, decode_seq_len, args.dim), dtype=torch.bfloat16
    )

    # Pre-populate caches with random data to simulate previous prefill phase
    attention.kv_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.kv_lora_rank, dtype=torch.bfloat16
    )
    attention.pe_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.qk_rope_head_dim, dtype=torch.bfloat16
    )
    attention.indexer.k_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.index_head_dim, dtype=torch.bfloat16
    )

    # Prepopulating topk_indices instead of running the indexer, since we have no
    # guarantee that the topk indices returned by it will be the same across CPU and
    # TT devices. Also, the indexer is already separately tested.
    # attention.prepopulated_topk_indices = torch.stack(
    #     [torch.randperm(prefill_seq_len)[: args.index_topk] for _ in range(batch_size)]
    # ).unsqueeze(1)  # (batch_size, 1, index_topk)

    # Get rotary embeddings for the current position
    freqs_cis = model.freqs_cis[start_pos : start_pos + decode_seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(attention, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        # Conditionally shard weights that involve batch axis
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        # Input tensors
        # shard_specs[args[0]] = (None, None, batch_axis)  # hidden_states (batch, 1, dim)
        shard_specs[args[0]] = (None, None, None)  # hidden_states (batch, 1, dim)

        # Weight tensors
        shard_specs[attention.wq_b.weight] = ("model", None)
        shard_specs[attention.wkv_b.weight] = ("model", None)
        shard_specs[attention.wo.weight] = (batch_axis, "model")
        shard_specs[attention.wq_a.weight] = (None, batch_axis)
        shard_specs[attention.wkv_a.weight] = (None, batch_axis)

        # Cache tensors
        shard_specs[attention.kv_cache] = (batch_axis, None, None)
        shard_specs[attention.pe_cache] = (batch_axis, None, None)

        # Indexer sharding (if present)
        shard_specs[attention.indexer.wq_b.weight] = ("model", None)
        shard_specs[attention.indexer.wk.weight] = (None, batch_axis)
        shard_specs[attention.indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[attention.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    # Force the attention output (batch, 1, dim) to be fully replicated across all
    # devices. Without this, wo.weight=(batch_axis, "model") leaves the dim axis
    # split across the two batch groups; the hook inserts an all-gather on the
    # batch axis so every device holds the complete (batch, 1, dim) tensor.
    hook_handle = attention.wo.register_forward_hook(
        sharding_constraint_hook(attention.wo, mesh, (None, None, None))
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            start_pos,
            freqs_cis,
            None,  # attention_mask - triggers decode path
            True,  # use_optimized_decode_flow
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )
    hook_handle.remove()


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 128, 512])
def test_deepseek_indexer(batch_size, seq_len):
    xr.set_device_type("TT")

    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

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
        pcc=PccConfig(enabled=True, required_pcc=0.99),
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


@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("prefill_seq_len", [32, 128, 512, 2048])
def test_dsa_optimized_decode_flow_compared_to_original(batch_size, prefill_seq_len):
    """
    This test compares the optimized decode flow with the original reference flow provided
    by Deepseek. It is run only on CPU.
    """
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        index_topk=16,
    )

    modified_model = ModifiedTransformer(args)
    modified_model = modified_model.to(torch.bfloat16)
    attention = modified_model.layers[0].attn

    freqs_cis = modified_model.freqs_cis[start_pos : start_pos + decode_seq_len]

    hidden_states = torch.randn(
        (batch_size, decode_seq_len, args.dim), dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        batch_size, start_pos, args.kv_lora_rank, dtype=torch.bfloat16
    )
    pe_cache = torch.randn(
        batch_size, start_pos, args.qk_rope_head_dim, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        batch_size, start_pos, args.index_head_dim, dtype=torch.bfloat16
    )
    end_pos = start_pos + decode_seq_len
    topk_indices = torch.stack(
        [torch.randperm(end_pos)[: args.index_topk] for _ in range(batch_size)]
    ).unsqueeze(
        1
    )  # (batch_size, 1, index_topk)

    attention.prepopulated_topk_indices = topk_indices
    attention.kv_cache[:batch_size, :start_pos] = kv_cache
    attention.pe_cache[:batch_size, :start_pos] = pe_cache
    attention.indexer.k_cache[:batch_size, :start_pos] = k_cache

    test_modified_output = attention(
        hidden_states, start_pos, freqs_cis, mask=None, use_optimized_decode_flow=True
    )
    test_original_output = attention(
        hidden_states, start_pos, freqs_cis, mask=None, use_optimized_decode_flow=False
    )

    pcc = compute_pcc(test_modified_output, test_original_output)

    assert pcc > 0.99, f"PCC too low: {pcc}"


def _map_hf_key_to_custom(hf_key):
    """Map a HuggingFace checkpoint key to the corresponding modified_model.py parameter name."""
    import re

    patterns = [
        (r"^model\.embed_tokens\.weight$", "embed.weight"),
        (r"^model\.norm\.weight$", "norm.weight"),
        (r"^lm_head\.weight$", "head.weight"),
        (
            r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$",
            r"layers.\1.attn.wq_a.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.q_a_layernorm\.weight$",
            r"layers.\1.attn.q_norm.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$",
            r"layers.\1.attn.wq_b.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$",
            r"layers.\1.attn.wkv_a.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.weight$",
            r"layers.\1.attn.kv_norm.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$",
            r"layers.\1.attn.wkv_b.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",
            r"layers.\1.attn.wo.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.input_layernorm\.weight$",
            r"layers.\1.attn_norm.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$",
            r"layers.\1.ffn_norm.weight",
        ),
        # Dense MLP (first n_dense_layers)
        (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"layers.\1.ffn.w1.weight"),
        (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"layers.\1.ffn.w2.weight"),
        (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"layers.\1.ffn.w3.weight"),
        # MoE gate
        (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$", r"layers.\1.ffn.gate.weight"),
        (
            r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$",
            r"layers.\1.ffn.gate.bias",
        ),
        # MoE routed experts
        (
            r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$",
            r"layers.\1.ffn.experts.\2.w1.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$",
            r"layers.\1.ffn.experts.\2.w2.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$",
            r"layers.\1.ffn.experts.\2.w3.weight",
        ),
        # MoE shared experts
        (
            r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$",
            r"layers.\1.ffn.shared_experts.w1.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$",
            r"layers.\1.ffn.shared_experts.w2.weight",
        ),
        (
            r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$",
            r"layers.\1.ffn.shared_experts.w3.weight",
        ),
    ]

    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, hf_key)
        if result != hf_key:
            return result

    return None  # Key not used by this model


def _load_hf_weights_into_model(model, model_path):
    """
    Load weights from HuggingFace safetensors shards into the modified_model.py Transformer.

    Iterates over all shards from the index file, maps HuggingFace key names to the
    custom naming convention, and loads with strict=False (the Indexer's weights are
    not present in the HuggingFace checkpoint and remain randomly initialized).
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    model_state = model.state_dict()
    custom_state = {}

    for shard_file in sorted(set(weight_map.values())):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in f.keys():
                custom_key = _map_hf_key_to_custom(hf_key)
                if custom_key is None or custom_key not in model_state:
                    continue
                tensor = f.get_tensor(hf_key)
                # Cast to match the dtype the model parameter was initialized with
                target_dtype = model_state[custom_key].dtype
                custom_state[custom_key] = tensor.to(target_dtype)

    missing, _ = model.load_state_dict(custom_state, strict=False)

    non_indexer_missing = [k for k in missing if "indexer" not in k]
    assert (
        not non_indexer_missing
    ), f"Unexpected missing keys after weight loading: {non_indexer_missing}"


def test_run_modified_model_e2e():
    """
    End-to-end test that downloads real DeepSeek V3.2 weights from HuggingFace,
    loads them into the modified Transformer, and runs a prefill step followed
    by a single decode step on TT hardware.

    The Indexer component (DSA-specific) is disabled here because its weights are
    not present in the HuggingFace checkpoint.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

    model_path = snapshot_download(repo_id=MODEL_NAME)

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = hf_config.rope_scaling or {}

    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=2048,
        vocab_size=hf_config.vocab_size,
        dim=hf_config.hidden_size,
        inter_dim=hf_config.intermediate_size,
        moe_inter_dim=hf_config.moe_intermediate_size,
        n_layers=hf_config.num_hidden_layers,
        n_dense_layers=hf_config.first_k_dense_replace,
        n_heads=hf_config.num_attention_heads,
        n_routed_experts=hf_config.n_routed_experts,
        n_shared_experts=hf_config.n_shared_experts,
        n_activated_experts=hf_config.num_experts_per_tok,
        n_expert_groups=hf_config.n_group,
        n_limited_groups=hf_config.topk_group,
        score_func=hf_config.scoring_func,
        route_scale=hf_config.routed_scaling_factor,
        q_lora_rank=hf_config.q_lora_rank,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        rope_theta=hf_config.rope_theta,
        rope_factor=rope_scaling.get("factor", 40),
        beta_fast=int(rope_scaling.get("beta_fast", 32)),
        beta_slow=int(rope_scaling.get("beta_slow", 1)),
        mscale=rope_scaling.get("mscale", 1.0),
        original_seq_len=rope_scaling.get("original_max_position_embeddings", 4096),
        # Indexer weights are not in the HuggingFace checkpoint; disable it here.
        index_n_heads=0,
    )

    model = ModifiedTransformer(args)
    _load_hf_weights_into_model(model, model_path)
    model = model.to(torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]  # (1, seq_len)

    xr.set_device_type("TT")
    compiled_model = torch.compile(model, backend="tt")
    device = torch_xla.device()
    compiled_model = compiled_model.to(device)
    tokens = tokens.to(device)

    # Prefill: process the full prompt
    with torch.no_grad():
        logits = compiled_model(tokens)
        logits = logits.to("cpu")

    assert logits.shape == (
        1,
        args.vocab_size,
    ), f"Unexpected prefill logits shape: {logits.shape}"

    # Decode: generate one new token from the last predicted token
    next_token = logits.argmax(dim=-1, keepdim=True).to(device)  # (1, 1)
    prefill_len = tokens.shape[1]

    with torch.no_grad():
        decode_logits = compiled_model(next_token, start_pos=prefill_len)
        decode_logits = decode_logits.to("cpu")

    assert decode_logits.shape == (
        1,
        args.vocab_size,
    ), f"Unexpected decode logits shape: {decode_logits.shape}"

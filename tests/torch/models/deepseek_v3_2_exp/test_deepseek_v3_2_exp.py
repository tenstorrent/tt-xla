# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.testers.compiler_config import CompilerConfig
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation

# This model is modified from the original deepseek_v3_2_exp model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
#    - fast_hadamard_transform requires a CUDA enviroment and fails to install
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
#    - the original implementation (kernel.py) relies on custom tilelang kernels not supported on TT
# 3. Avoid torch.view_as_complex/view_as_real operations


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
def test_deepseek_attention_prefill(batch_size):
    xr.set_device_type("TT")
    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
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


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_indexer(batch_size):
    xr.set_device_type("TT")

    seq_len = 32
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
def test_deepseek_v3_2_layer_sparse_moe():
    """Test single MoE Block with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    import resource

    def peak_rss_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    batch_size = 64
    seq_len = 1
    args = ModelArgs(
        n_layers=2,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    # Create full model to get freqs_cis, then extract MoE block (layer 1)
    print(f"[mem] before model init: {peak_rss_gb():.2f} GB")
    model = ModifiedTransformer(args)
    print(f"[mem] after model init: {peak_rss_gb():.2f} GB")
    model = model.to(torch.bfloat16)
    print(f"[mem] after to(bf16): {peak_rss_gb():.2f} GB")
    block = model.layers[1]  # layer_id=1 >= n_dense_layers=1 → MoE
    freqs_cis = model.freqs_cis[:seq_len]

    mesh_shape = (4, 8)
    print(f"[mem] before enable_sparse_mlp: {peak_rss_gb():.2f} GB")
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    print(f"[mem] after enable_sparse_mlp: {peak_rss_gb():.2f} GB")
    block.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    print(f"[mem] before run_graph_test: {peak_rss_gb():.2f} GB")

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
        shard_specs[mlp.experts.gate_up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_up_proj_bias] = (
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
    print(f"[mem] after run_graph_test: {peak_rss_gb():.2f} GB")


@pytest.mark.llmbox
def test_deepseek_v3_2_full_sparse_moe():
    """Test full DeepseekV3-2 Transformer with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    import resource

    def peak_rss_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    batch_size = 16
    seq_len = 32
    args = ModelArgs(
        # n_layers=2,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    print(f"[mem] before model init: {peak_rss_gb():.2f} GB")
    model = ModifiedTransformer(args)
    print(f"[mem] after model init: {peak_rss_gb():.2f} GB")
    model = model.to(torch.bfloat16)
    print(f"[mem] after to(bf16): {peak_rss_gb():.2f} GB")
    # head is intentionally float32 in the original model (logits computed in fp32),
    # but model.to(bf16) converts it. Restore to float32 to match forward's .float() call.
    model.head = model.head.to(torch.float32)

    mesh_shape = (4, 8)
    print(f"[mem] before enable_sparse_mlp: {peak_rss_gb():.2f} GB")
    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=args,
    )
    print(f"[mem] after enable_sparse_mlp: {peak_rss_gb():.2f} GB")

    model.eval()

    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    # Pre-compute freqs_cis and mask outside forward() to avoid an unsharded
    # graph segment that compiles before the device mesh is opened with the
    # correct shape.  Without this, the first compiled segment has a trivial
    # [1,1] shardy mesh that falls back to a wrong [1,N] layout and segfaults.
    freqs_cis = model.freqs_cis[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    print(f"[mem] before run_graph_test: {peak_rss_gb():.2f} GB")

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
                shard_specs[mlp.experts.gate_up_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.gate_up_proj_bias] = (
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
        [tokens, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )
    print(f"[mem] after run_graph_test: {peak_rss_gb():.2f} GB")

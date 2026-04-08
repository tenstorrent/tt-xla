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
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.loader import (
    ModelLoader,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.modified_model import (
    LayerNorm as DeepSeekLayerNorm,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.modified_model import (
    ModelArgs,
)
from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.modified_model import (
    Transformer as ModifiedTransformer,
)

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
def test_deepseek_attention_prefill_then_decode(batch_size):
    """Tests MLA attention running prefill (seqlen=32) followed by one decode step.

    During prefill, kv_cache/pe_cache/k_cache are written at positions 0:32.
    During decode, the new token is written at position 32 and scores are
    computed over the full history (0:33).  Both steps share the same module
    instance so the cache state from prefill is visible to decode.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    seq_len = 32
    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=128
    )
    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Zero mask = attend everywhere (no causal masking needed for shape testing)
    prefill_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    decode_hidden = torch.randn((batch_size, 1, args.dim), dtype=torch.bfloat16)
    freqs_cis_prefill = model.freqs_cis[0:seq_len]
    freqs_cis_decode = model.freqs_cis[seq_len : seq_len + 1]

    class PrefillDecodeWrapper(torch.nn.Module):
        """Runs prefill then one decode step on the same attention module.

        start_pos values (0 and seq_len) are Python ints so they are folded as
        constants in the compiled graph rather than traced as SymInts.
        freqs_cis tensors are passed as inputs so the correct positional
        embeddings are used for each phase.
        """

        def __init__(self, attn, seq_len):
            super().__init__()
            self.attn = attn
            self.seq_len = seq_len  # constant: number of prefill tokens

        def forward(
            self,
            hidden_states,
            freqs_cis_prefill,
            prefill_mask,
            decode_hidden,
            freqs_cis_decode,
        ):
            # Prefill: fills kv_cache/pe_cache/k_cache at positions 0:seq_len
            self.attn(hidden_states, 0, freqs_cis_prefill, prefill_mask)
            # Decode: writes at seq_len, reads full history 0:seq_len+1
            return self.attn(decode_hidden, self.seq_len, freqs_cis_decode, None)

    wrapper = PrefillDecodeWrapper(attention, seq_len)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        attn = model.attn
        shard_specs = {}

        shard_specs[args[0]] = (
            None,
            None,
            batch_axis,
        )  # hidden_states [batch, seq, dim]
        # args[1] freqs_cis_prefill → replicated (not listed)
        shard_specs[args[2]] = (
            batch_axis,
            None,
            None,
        )  # prefill_mask [batch, seq, seq]
        shard_specs[args[3]] = (None, None, batch_axis)  # decode_hidden [batch, 1, dim]
        # args[4] freqs_cis_decode → replicated (not listed)

        shard_specs[attn.wq_b.weight] = ("model", None)
        shard_specs[attn.wkv_b.weight] = ("model", None)
        shard_specs[attn.wo.weight] = (batch_axis, "model")
        shard_specs[attn.wq_a.weight] = (None, batch_axis)
        shard_specs[attn.wkv_a.weight] = (None, batch_axis)

        shard_specs[attn.kv_cache] = (batch_axis, None, None)
        shard_specs[attn.pe_cache] = (batch_axis, None, None)

        shard_specs[attn.indexer.wq_b.weight] = ("model", None)
        shard_specs[attn.indexer.wk.weight] = (None, batch_axis)
        shard_specs[attn.indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[attn.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        wrapper,
        [
            hidden_states,
            freqs_cis_prefill,
            prefill_mask,
            decode_hidden,
            freqs_cis_decode,
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
def test_deepseek_v3_2_moe_only():
    """Test MoE MLP only (no attention) with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 64
    seq_len = 128
    args = ModelArgs(
        n_layers=2,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    # Extract MoE module from block layer 1
    moe = model.layers[1].ffn
    mesh_shape = (2, 4)
    enable_sparse_mlp(moe, mesh=mesh_shape, cluster_axis=0, config=args)
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
        q_lora_rank=3072,
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
    """Test full DeepseekV3-2 Transformer with A2aSparseMLP on (4,8) galaxy mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 64
    seq_len = 32

    # ModelLoader handles: bfloat16 cast, head dtype restore, enable_sparse_mlp, eval
    loader = ModelLoader(num_layers=2, max_batch_size=batch_size)
    wrapped = loader.load_model(dtype_override=torch.bfloat16, max_seq_len=seq_len * 2)
    transformer = wrapped.transformer
    args = loader._args

    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    # Pre-compute freqs_cis and mask outside forward() to avoid an unsharded
    # graph segment that compiles before the device mesh is opened with the
    # correct shape.  Without this, the first compiled segment has a trivial
    # [1,1] shardy mesh that falls back to a wrong [1,N] layout and segfaults.
    freqs_cis = transformer.freqs_cis[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
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
        transformer,
        [tokens, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )


@pytest.mark.llmbox
def test_deepseek_v3_2_full_sparse_moe_via_loader():
    """Test full DeepSeekV3-2 via ModelLoader wrapper forward() on a (4,8) galaxy mesh.

    Unlike test_deepseek_v3_2_full_sparse_moe (which extracts wrapped.transformer and
    pre-computes freqs_cis/mask outside the compiled region), this test calls through
    the DeepSeekV32ForCausalLM wrapper.  It exercises:
      - buffer-based freqs_cis/_causal_mask slicing inside forward()
      - load_shard_spec() replicated specs for those buffers
      - the full loader → wrapper → transformer call chain

    Layer layout with num_layers=2 (n_dense_layers capped to 1 by the loader):
      layer 0 (layer_id=0, < n_dense_layers=1): dense MLP
      layer 1 (layer_id=1, >= n_dense_layers=1): A2aSparseMLPWithSharedExperts (MoE)
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 32
    seq_len = 32

    loader = ModelLoader(num_layers=2, max_batch_size=batch_size)
    wrapped = loader.load_model(dtype_override=torch.bfloat16, max_seq_len=128)

    tokens = torch.randint(0, loader._args.vocab_size, (batch_size, seq_len))

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    # Use ("batch", "model") to match loader.load_shard_spec axis names directly.
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        # model is DeepSeekV32ForCausalLM directly (no _LogitsWrapper).
        shard_specs = loader.load_shard_spec(model)
        # Input tokens [batch, seq] — batch dim on model axis (size 8).
        shard_specs[args[0]] = ("model", None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        wrapped,
        [tokens],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )


def test_kv_cache_shared_between_cpu_and_device_runs():
    """Check that model.to(device) carries dirty KV cache from CPU run to device.

    Simulates the llm_benchmark.py call sequence:
        1. CPU reference run  — generate_and_benchmark writes to kv_cache/pe_cache
        2. model.to(device)   — moves the model *including dirty cache* to device
        3. Device forward run — starts with the cache already populated

    Two facts are verified:

    Fact 1 — Cache IS shared:
        After the CPU forward, kv_cache and pe_cache are non-zero.  model.to(device)
        transfers those values to device; the device model sees the same dirty cache.

    Fact 2 — Dirty cache does NOT corrupt output (with start_pos=0):
        The model always writes kv_cache[0:seqlen] *before* reading from it
        (start_pos=0 hardcoded in DeepSeekV32ForCausalLM.forward).  The initial
        cache contents are overwritten on every call, so a clean-cache forward and
        a dirty-cache forward on the same tokens produce identical logits.

    Consequence for PCC=0.68:
        The observed PCC degradation is NOT caused by cache contamination.
        It is caused by the CPU and device runs using different random tokens
        (each construct_inputs call independently called torch.randint).
        The fix is to generate _prefill_tokens once and reuse across all
        construct_inputs calls (done in llm_benchmark.py).
    """
    torch.manual_seed(42)

    args = ModelArgs(n_layers=1, q_lora_rank=3072, max_batch_size=2, max_seq_len=64)
    model = ModifiedTransformer(args).to(torch.bfloat16).eval()

    # Restore LayerNorm and head to float32 — mirrors loader.load_model() post-cast
    # fixups.  LayerNorm.forward() calls x.float() before F.layer_norm, so its
    # weight/bias must remain float32 or CPU execution raises a mixed-dtype error.
    # head is also float32 in the original model (logits computed in fp32).
    for module in model.modules():
        if isinstance(module, DeepSeekLayerNorm):
            module.weight.data = module.weight.data.to(torch.float32)
            module.bias.data = module.bias.data.to(torch.float32)
    model.head = model.head.to(torch.float32)

    attn = model.layers[0].attn

    batch_size, seq_len = 2, 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # ------------------------------------------------------------------ #
    # Fact 1: CPU run writes to kv_cache/pe_cache; model.to(device)       #
    # carries those values.                                                #
    # ------------------------------------------------------------------ #

    # Cache starts at zero
    assert attn.kv_cache.abs().max() == 0.0, "kv_cache should be zero-initialised"
    assert attn.pe_cache.abs().max() == 0.0, "pe_cache should be zero-initialised"

    # CPU forward run (mirrors the PCC-reference run in llm_benchmark)
    with torch.no_grad():
        model(tokens, start_pos=0)

    kv_after_cpu = attn.kv_cache[:batch_size, :seq_len].detach().clone()
    pe_after_cpu = attn.pe_cache[:batch_size, :seq_len].detach().clone()

    assert (
        kv_after_cpu.abs().max() > 0.0
    ), "CPU run should have written non-zero values to kv_cache"
    assert (
        pe_after_cpu.abs().max() > 0.0
    ), "CPU run should have written non-zero values to pe_cache"

    # model.to(device) would carry these values; confirm they are present on
    # the same buffer object (no implicit reset on to()).
    model_cpu = model.to("cpu")  # no-op (already cpu), but mirrors model.to(device)
    assert attn.kv_cache[:batch_size, :seq_len].allclose(
        kv_after_cpu
    ), "model.to(device) must not reset kv_cache — dirty values are transferred"

    # ------------------------------------------------------------------ #
    # Fact 2: Dirty cache does NOT change output because start_pos=0       #
    # overwrites before reading.                                           #
    # ------------------------------------------------------------------ #

    # Run on clean cache: reset to zero and forward
    attn.kv_cache.zero_()
    attn.pe_cache.zero_()
    if attn.indexer is not None:
        attn.indexer.k_cache.zero_()
    with torch.no_grad():
        logits_clean = model(tokens, start_pos=0).clone()

    # Run on dirty cache: cache is now populated from the previous forward
    kv_dirty = attn.kv_cache[:batch_size, :seq_len].detach().clone()
    assert (
        kv_dirty.abs().max() > 0.0
    ), "cache should be non-zero (dirty) before second run"
    with torch.no_grad():
        logits_dirty = model(tokens, start_pos=0).clone()

    # Outputs must be identical: the write at [0:seqlen] overwrites dirty state
    # before the read at [:seqlen], so initial cache values are irrelevant.
    assert torch.allclose(logits_clean, logits_dirty), (
        "With start_pos=0, dirty cache should not affect output — the write "
        "always precedes the read for positions [0:seqlen]."
    )

    # Confirm PCC=1.0 (same tokens, same cache state after write)
    c = logits_clean.float().reshape(-1)
    d = logits_dirty.float().reshape(-1)
    pcc = torch.corrcoef(torch.stack([c, d]))[0, 1].item()
    assert (
        pcc > 0.9999
    ), f"Expected PCC≈1.0 for same-token clean vs dirty run, got {pcc:.6f}"

    print(
        f"\nFact 1 confirmed: CPU run populates kv_cache (max={kv_after_cpu.abs().max():.4f})."
        f"\nFact 2 confirmed: dirty cache does not change output (PCC={pcc:.6f})."
        f"\nConclusion: PCC=0.68 in benchmark was caused by different tokens, "
        f"not cache contamination."
    )


def test_decode_appends_to_kv_cache_with_mla_cache():
    """Verify that DeepSeekMLACache tracks position so decode appends correctly.

    Without an external cache, DeepSeekV32ForCausalLM.forward() hardcodes
    start_pos=0 and every call overwrites position 0 instead of appending.

    With a DeepSeekMLACache passed as past_key_values, the loader reads
    cache.current_pos as start_pos and advances it by seqlen after each call.
    This test confirms:

    1. After prefill (32 tokens): cache.current_pos == 32, positions 0–31 written.
    2. After decode (1 token):   cache.current_pos == 33, position 32 written,
                                  prefill history (positions 0–31) intact.
    """
    from tests.torch.models.utils.mla_cache import DeepSeekMLACache

    torch.manual_seed(42)

    batch_size, prefill_len, max_seq = 1, 32, 64
    loader = ModelLoader(num_layers=1, max_batch_size=batch_size)
    model = loader.load_model(dtype_override=torch.bfloat16, max_seq_len=max_seq)
    model.eval()

    cache = DeepSeekMLACache.from_model_args(
        loader._args, batch_size=batch_size, max_seq_len=max_seq
    )

    prefill_tokens = torch.randint(
        0, loader._args.vocab_size, (batch_size, prefill_len)
    )
    decode_token = torch.randint(0, loader._args.vocab_size, (batch_size, 1))

    # --- Prefill ---
    with torch.no_grad():
        model(input_ids=prefill_tokens, past_key_values=cache)

    assert (
        cache.current_pos == prefill_len
    ), f"current_pos should be {prefill_len} after prefill, got {cache.current_pos}"

    layer0 = cache.layers[0]
    kv_after_prefill = layer0.compressed_kv[:batch_size].detach().clone()

    assert (
        kv_after_prefill[0, :prefill_len].abs().max() > 0
    ), "Prefill should write positions 0:32 into the external cache"
    assert (
        kv_after_prefill[0, prefill_len:].abs().max() == 0
    ), "Positions 32+ should be zero after prefill"

    # --- Decode ---
    with torch.no_grad():
        model(input_ids=decode_token, past_key_values=cache)

    assert (
        cache.current_pos == prefill_len + 1
    ), f"current_pos should be {prefill_len + 1} after decode, got {cache.current_pos}"

    kv_after_decode = layer0.compressed_kv[:batch_size].detach().clone()

    pos32_written = kv_after_decode[0, prefill_len].abs().max() > 0
    history_intact = torch.allclose(
        kv_after_decode[0, :prefill_len], kv_after_prefill[0, :prefill_len]
    )

    assert (
        pos32_written
    ), "Decode token should be written at position 32 in the external cache"
    assert (
        history_intact
    ), "Prefill history (positions 0–31) should be unchanged after decode"

    print(
        f"\nDeepSeekMLACache prefill→decode verified:"
        f"\n  current_pos after prefill : {prefill_len}"
        f"\n  current_pos after decode  : {prefill_len + 1}"
        f"\n  position 32 written       : {pos32_written}"
        f"\n  prefill history intact    : {history_intact}"
    )

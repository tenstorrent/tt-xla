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
from original_modified_model import Transformer as OriginalModifiedTransformer
from original_modified_model import precompute_freqs_cis
from torch_xla.distributed.spmd import Mesh

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


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.constant'"
    )
)
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

    compiled_apply_rotary_emb = torch.compile(apply_rotary_emb, backend="tt")

    batch_size = 2
    seq_len = 16
    dim = 64
    n_heads = 4
    head_dim = dim // n_heads

    x = torch.randn(
        batch_size, seq_len, n_heads, head_dim, device="xla", dtype=torch.bfloat16
    )
    freqs_cis = torch.randn(seq_len, head_dim // 2, device="xla", dtype=torch.complex64)

    y = compiled_apply_rotary_emb(x, freqs_cis, interleaved=True)
    assert y.shape == x.shape


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
def test_deepseek_attention_decode(batch_size):
    xr.set_device_type("TT")

    # Decode-specific parameters
    prefill_seq_len = 10  # Simulate that 32 tokens were already processed
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        # index_topk=16
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

    # Get rotary embeddings for the current position
    freqs_cis = model.freqs_cis[start_pos : start_pos + decode_seq_len]

    # attention_mask=None triggers the decode path (MQA)
    attention_mask = None

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
        shard_specs[args[0]] = (None, None, batch_axis)  # hidden_states (batch, 1, dim)

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

    comparison_config = ComparisonConfig(
        pcc=PccConfig(
            enabled=True, required_pcc=0.88
        ),  # Allow for BF16 precision limits
    )

    run_graph_test(
        attention,
        [
            hidden_states,  # (batch_size, 1, dim)
            start_pos,  # 32 (or prefill_seq_len)
            freqs_cis,  # (1, qk_rope_head_dim)
            attention_mask,  # None - triggers decode path
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_attention_decode_flow_compared_to_original(batch_size):
    xr.set_device_type("TT")

    # Decode-specific parameters
    prefill_seq_len = 10  # Simulate that 32 tokens were already processed
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        # index_topk=16
    )

    modified_model = ModifiedTransformer(args)
    original_model = OriginalModifiedTransformer(args)
    modified_model = modified_model.to(torch.bfloat16)
    original_model = original_model.to(torch.bfloat16)
    modified_attention = modified_model.layers[0].attn
    original_attention = original_model.layers[0].attn

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

    original_attention.kv_cache[:batch_size, :start_pos] = kv_cache
    original_attention.pe_cache[:batch_size, :start_pos] = pe_cache
    original_attention.indexer.k_cache[:batch_size, :start_pos] = k_cache
    modified_attention.kv_cache[:batch_size, :start_pos] = kv_cache
    modified_attention.pe_cache[:batch_size, :start_pos] = pe_cache
    modified_attention.indexer.k_cache[:batch_size, :start_pos] = k_cache

    freq_cis_original = original_model.freqs_cis[start_pos : start_pos + decode_seq_len]
    freq_cis_modified = modified_model.freqs_cis[start_pos : start_pos + decode_seq_len]

    assert torch.allclose(
        freq_cis_original, freq_cis_modified
    ), "freq_cis_original and freq_cis_modified are not the same"
    attention_mask = None

    test_modified_output = modified_attention(
        hidden_states, start_pos, freq_cis_modified, attention_mask
    )
    test_original_output = original_attention(
        hidden_states, start_pos, freq_cis_original, attention_mask
    )

    pcc = compute_pcc(test_modified_output, test_original_output)
    print(f"PCC between modified and original: {pcc.item()}")

    # Sanity check: outputs should be numerically close (not required to be exact, but at least testable for basic code correctness)
    assert torch.isfinite(
        test_modified_output
    ).all(), "Modified output contains non-finite values"
    assert torch.isfinite(
        test_original_output
    ).all(), "Original output contains non-finite values"
    assert pcc > 0.99, f"PCC too low: {pcc.item()}"


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_deepseek_indexer(batch_size):
    xr.set_device_type("TT")

    seq_len = 32
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_topk=16,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    indexer = model.layers[0].attn.indexer

    # Enable raw score return for testing (returns index_score instead of topk_indices)
    indexer.return_raw_scores = False

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


def topk_indices_comparator(device_output, golden_output, inputs):
    device_values, device_indices = device_output
    golden_values, _ = golden_output
    input_tensor = inputs[0]

    # Move device outputs from XLA to CPU for comparison
    device_values = device_values.cpu()
    device_indices = device_indices.cpu()

    # 1) PCC between golden_values and device_values
    pcc = compute_pcc(golden_values, device_values)
    assert pcc > 0.99, f"PCC between golden and device values: {pcc} (required > 0.99)"

    # 2) Assert device_indices has no duplicate elements (per row, along last dim)
    for i in range(device_indices.shape[0]):
        row = device_indices[i]
        assert (
            row.unique().numel() == row.numel()
        ), "Duplicate indices found in device output"

    # 3) Gather values using device_indices, compute cosine similarity with golden_values
    gathered = torch.gather(input_tensor, -1, device_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        gathered.flatten().unsqueeze(0).float(),
        golden_values.flatten().unsqueeze(0).float(),
    )
    print(f"Cosine similarity: {cos_sim}")
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


def test_topk():
    xr.set_device_type("TT")

    class TopKIndices(torch.nn.Module):
        def forward(self, x):
            output, indices = torch.topk(x, k=64, dim=-1)
            return output, indices

    model = TopKIndices()
    # input_tensor = torch.arange(64, 0, -1, dtype=torch.bfloat16).view(1, 64)
    input_tensor = torch.randn(1, 64, dtype=torch.bfloat16)
    print(f"input_tensor: {input_tensor}")

    run_graph_test(
        model,
        [input_tensor],
        framework=Framework.TORCH,
        custom_comparator=topk_indices_comparator,
    )


def test_sparse_gather():
    xr.set_device_type("TT")

    topk = 16
    kv_lora_rank = 512

    class SparseGather(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_lora_rank = 512

        def forward(self, topk_indices, kv_cache):
            gather_idx = topk_indices.squeeze(1)  # (bsz, topk)

            kv_sparse = kv_cache.gather(
                1, gather_idx.unsqueeze(-1).expand(-1, -1, kv_lora_rank)
            )
            # kv_sparse = kv_cache.gather(
            #     1, gather_idx.)
            return kv_sparse

    model = SparseGather()
    kv_cache = torch.randn(1, topk, kv_lora_rank, dtype=torch.bfloat16)
    topk_indices = torch.randint(0, topk, (1, 1, topk), dtype=torch.int64)

    run_graph_test(
        model,
        [topk_indices, kv_cache],
        framework=Framework.TORCH,
    )

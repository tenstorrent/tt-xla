# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, MLACache, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers import DynamicCache
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation
from third_party.tt_forge_models.kimi_k2.pytorch.configuration_deepseek import (
    DeepseekV3Config,
)
from third_party.tt_forge_models.kimi_k2.pytorch.loader import ModelLoader
from third_party.tt_forge_models.kimi_k2.pytorch.modified_modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
)

from .original_modeling_deepseek import DeepseekV3Attention as OrigDeepseekV3Attention


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 32. "
    )
)
def test_kimi_k2_single_layer():
    xr.set_device_type("TT")

    loader = ModelLoader()
    config = loader._load_config(num_layers=1)
    config.use_cache = False

    model = DeepseekV3ForCausalLM(config)

    batch_size = 64
    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    model = model.to(torch.bfloat16)
    model = model.eval()

    compiled_model = torch.compile(model, backend="tt")

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.to("cpu")


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.lb_blackhole
def test_kimi_k2_attention_prefill():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader()
    config = loader._load_config(num_layers=1)

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    batch_size = 64
    seq_len = 32
    max_cache_len = 1024
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = (None, None, "batch")
        shard_specs[attention.q_b_proj.weight] = ("model", None)
        shard_specs[attention.kv_b_proj.weight] = ("model", None)
        shard_specs[attention.o_proj.weight] = ("batch", "model")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "batch")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "batch")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    # Diagnostic: capture attention output across runs. Run the test multiple
    # times and compare the printed hashes to determine whether attention itself
    # is non-deterministic. Stable hashes => attention is bit-reproducible; any
    # downstream variance originates after attention. Different hashes =>
    # attention contributes its own non-determinism.
    captured_attn_outputs = []

    def capture_attn_output(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        inp = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else None
        dev = str(inp.device) if (inp is not None and hasattr(inp, "device")) else "?"
        captured_attn_outputs.append((dev, out))

    hook_handle = attention.register_forward_hook(capture_attn_output)
    try:
        run_graph_test(
            attention,
            [
                hidden_states,
                attention_mask,
                position_ids,
                past_key_states,
                False,
                True,
                cache_positions,
            ],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=get_shard_spec,
            comparison_config=comparison_config,
        )
    finally:
        hook_handle.remove()

    print(
        f"\n[attention-prefill diagnostic] captured "
        f"{len(captured_attn_outputs)} forward call(s):"
    )
    for i, (dev, out) in enumerate(captured_attn_outputs):
        try:
            cpu_out = out.detach().cpu().contiguous()
            flat = cpu_out.flatten().float()
            h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
            print(
                f"  call#{i}: device={dev} shape={tuple(cpu_out.shape)} "
                f"sum={flat.sum().item():.4f} hash={h} "
                f"first5={[round(v, 6) for v in flat[:5].tolist()]}",
                flush=True,
            )
        except Exception as e:
            print(f"  call#{i}: device={dev} <materialization failed: {e}>", flush=True)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.lb_blackhole
def test_kimi_k2_attention_decode():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader()
    config = loader._load_config(num_layers=1)

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    for layer in static_cache.layers:
        layer.lazy_initialization(
            torch.zeros(batch_size, 1, 1, config.kv_lora_rank, dtype=torch.bfloat16),
            torch.zeros(
                batch_size, 1, 1, config.qk_rope_head_dim, dtype=torch.bfloat16
            ),
        )
    past_key_states = static_cache

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].k_pe] = ("_axis_1", None, None, None)

        # Main attention weights, TP across model and batch dimensions
        shard_specs[attention.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.o_proj.weight] = (None, "_axis_0")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "_axis_0")
        return shard_specs

    # Diagnostic: capture attention output across runs. Same purpose as the
    # prefill variant but exercising the decode-shape attention path.
    captured_attn_outputs = []

    def capture_attn_output(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        inp = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else None
        dev = str(inp.device) if (inp is not None and hasattr(inp, "device")) else "?"
        captured_attn_outputs.append((dev, out))

    hook_handle = attention.register_forward_hook(capture_attn_output)
    try:
        run_graph_test(
            attention,
            [
                hidden_states,
                attention_mask,
                position_ids,
                past_key_states,
                False,
                True,
                cache_positions,
            ],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=get_shard_spec,
        )
    finally:
        hook_handle.remove()

    print(
        f"\n[attention-decode diagnostic] captured "
        f"{len(captured_attn_outputs)} forward call(s):"
    )
    for i, (dev, out) in enumerate(captured_attn_outputs):
        try:
            cpu_out = out.detach().cpu().contiguous()
            flat = cpu_out.flatten().float()
            h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
            print(
                f"  call#{i}: device={dev} shape={tuple(cpu_out.shape)} "
                f"sum={flat.sum().item():.4f} hash={h} "
                f"first5={[round(v, 6) for v in flat[:5].tolist()]}",
                flush=True,
            )
        except Exception as e:
            print(f"  call#{i}: device={dev} <materialization failed: {e}>", flush=True)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.lb_blackhole
def test_kimi_k2_layer():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader()
    config = loader._load_config(num_layers=1)
    config._attn_implementation = "eager"

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer = layer.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    position_ids = torch.arange(seq_len).unsqueeze(0)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    for cache_layer in static_cache.layers:
        cache_layer.lazy_initialization(
            torch.zeros(batch_size, 1, 1, config.kv_lora_rank, dtype=torch.bfloat16),
            torch.zeros(
                batch_size, 1, 1, config.qk_rope_head_dim, dtype=torch.bfloat16
            ),
        )
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].k_pe] = ("_axis_1", None, None, None)

        # Main attention weights, TP across model and batch dimensions
        shard_specs[layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "_axis_0")

        # Consume hidden states, TP on batch dimension
        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "_axis_0")

        shard_specs[layer.mlp.gate_proj.weight] = ("_axis_1", "_axis_0")
        shard_specs[layer.mlp.up_proj.weight] = ("_axis_1", "_axis_0")
        shard_specs[layer.mlp.down_proj.weight] = ("_axis_0", "_axis_1")

        shard_specs[layer.input_layernorm.weight] = ("_axis_0",)
        shard_specs[layer.post_attention_layernorm.weight] = ("_axis_0",)

        return shard_specs

    run_graph_test(
        layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_states,
            False,
            True,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [32])
def test_kimi_k2_layer_sparse_moe(batch_size, seq_len):
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader()
    config = loader._load_config(num_layers=2)
    config._attn_implementation = "eager"

    layer = DeepseekV3DecoderLayer(config, layer_idx=1)
    layer = layer.eval().to(torch.bfloat16)

    max_cache_len = 1024
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    enable_sparse_mlp(layer, mesh=mesh_shape, cluster_axis=0, config=config)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[1].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[1].k_pe] = ("_axis_1", None, None, None)

        shard_specs[layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "_axis_0")

        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "_axis_0")

        # A2aSparseMLP: experts compound-sharded (axis_0, axis_1)
        mlp_wrapper = layer.mlp
        mlp = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
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

        # Shared experts (if present, on wrapper not on inner A2aSparseMLP)
        shared = getattr(mlp_wrapper, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.gate_proj.weight] = (None, "_axis_0")
            shard_specs[shared.up_proj.weight] = (None, "_axis_0")
            shard_specs[shared.down_proj.weight] = ("_axis_0", None)

        # Layernorm: hidden replicated → weight not sharded
        shard_specs[layer.input_layernorm.weight] = ("_axis_0",)
        shard_specs[layer.post_attention_layernorm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.98),
    )

    # Diagnostic: capture the router's topk indices for each forward call.
    # Across separate pytest invocations, compare the printed hashes to tell
    # whether MoE routing decisions are stable run-to-run. Different hashes
    # for the same device path => routing is non-deterministic.
    mlp_wrapper = layer.mlp
    mlp_inner = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
    captured_router_outputs = []

    def capture_router_topk(module, inputs, output):
        indices = output[-1] if isinstance(output, tuple) else output
        inp = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else None
        dev = str(inp.device) if (inp is not None and hasattr(inp, "device")) else "?"
        captured_router_outputs.append((dev, indices))

    # Diagnostic: snapshot a hash of every layer parameter ONCE before
    # run_graph_test, while params are still on CPU. Run pytest several
    # times and compare these hashes alongside the router-topk hashes —
    # this disambiguates whether topk variance is caused by weights
    # changing run-to-run or by nondeterminism in routing for fixed
    # weights. (Snapshotting inside a forward_pre_hook caused device OOM
    # because it forced re-materialization of sharded XLA params.)
    def _hash_layer_params(module):
        per_param = []
        h_all = hashlib.sha256()
        for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
            t = p.detach().cpu().contiguous()
            # bfloat16 has no numpy dtype; view raw bytes through int16
            # (same element size) so we hash without a float32 copy.
            if t.dtype == torch.bfloat16:
                raw = t.view(torch.int16).numpy().tobytes()
            else:
                raw = t.numpy().tobytes()
            digest = hashlib.sha256(raw).hexdigest()[:16]
            per_param.append((name, tuple(p.shape), digest))
            h_all.update(name.encode())
            h_all.update(digest.encode())
        return h_all.hexdigest()[:16], per_param

    pre_run_agg, pre_run_per_param = _hash_layer_params(layer)

    hook_handle = mlp_inner.router.register_forward_hook(capture_router_topk)
    try:
        run_graph_test(
            layer,
            [
                hidden_states,
                attention_mask,
                position_ids,
                past_key_states,
                False,
                True,
                cache_positions,
            ],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=get_shard_spec,
            comparison_config=comparison_config,
        )
    finally:
        hook_handle.remove()

    print(
        f"\n[router topk diagnostic] captured {len(captured_router_outputs)} "
        f"router call(s):"
    )
    for i, (dev, idx) in enumerate(captured_router_outputs):
        try:
            cpu_idx = idx.detach().cpu().contiguous()
            flat = cpu_idx.flatten()
            h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
            print(
                f"  call#{i}: device={dev} shape={tuple(cpu_idx.shape)} "
                f"sum={int(flat.sum().item())} hash={h} "
                f"first10={flat[:10].tolist()}",
                flush=True,
            )
        except Exception as e:
            print(f"  call#{i}: device={dev} <materialization failed: {e}>", flush=True)

    print(f"\n[layer-params diagnostic] (snapshot before run_graph_test)")
    print(f"  agg={pre_run_agg}", flush=True)
    for name, shape, digest in pre_run_per_param:
        print(f"    {name} shape={shape} hash={digest}", flush=True)


@pytest.mark.nightly
def test_kimi_k2_mla_cache():
    """
    CPU-only test validating the MLACache used in modeling_deepseek.py against the original
    DynamicCache used in original_modeling_deepseek.py for DeepseekV3Attention.
    """

    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_hidden_layers=1,
        max_position_embeddings=64,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )

    BATCH_SIZE = 2
    PREFILL_LEN = 8
    LAYER_IDX = 0
    MAX_CACHE_LEN = PREFILL_LEN + 1

    mla_attn = DeepseekV3Attention(config, layer_idx=LAYER_IDX)
    mla_attn.eval()
    orig_attn = OrigDeepseekV3Attention(config, layer_idx=LAYER_IDX)
    orig_attn.load_state_dict(mla_attn.state_dict())
    orig_attn.eval()

    torch.manual_seed(0)

    # Prefill
    mla_cache = MLACache(config, max_cache_len=MAX_CACHE_LEN)
    orig_cache = DynamicCache()

    prefill_hidden = torch.randn(BATCH_SIZE, PREFILL_LEN, config.hidden_size)
    prefill_position_ids = torch.arange(PREFILL_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)

    mla_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, MAX_CACHE_LEN)
    orig_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, PREFILL_LEN)

    with torch.no_grad():
        mla_attn(
            prefill_hidden,
            mla_prefill_mask,
            prefill_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.arange(PREFILL_LEN),
        )
        orig_attn(
            prefill_hidden,
            orig_prefill_mask,
            prefill_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    # Decode
    decode_hidden = torch.randn(BATCH_SIZE, 1, config.hidden_size)
    decode_position_ids = torch.full((BATCH_SIZE, 1), PREFILL_LEN, dtype=torch.long)
    decode_mask = torch.zeros(BATCH_SIZE, 1, 1, MAX_CACHE_LEN)

    with torch.no_grad():
        mla_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.tensor([PREFILL_LEN]),
        )
        orig_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    total_len = PREFILL_LEN + 1

    orig_key = orig_cache.layers[LAYER_IDX].keys
    orig_val = orig_cache.layers[LAYER_IDX].values

    compressed_kv = mla_cache.layers[LAYER_IDX].compressed_kv[:, 0, :total_len, :]
    mla_k_pe = mla_cache.layers[LAYER_IDX].k_pe[:, :, :total_len, :]

    with torch.no_grad():
        kv = (
            mla_attn.kv_b_proj(mla_attn.kv_a_layernorm(compressed_kv))
            .view(
                -1,
                total_len,
                config.num_attention_heads,
                config.qk_nope_head_dim + config.v_head_dim,
            )
            .transpose(1, 2)
        )

    mla_k_nope, mla_val = torch.split(
        kv, [config.qk_nope_head_dim, config.v_head_dim], dim=-1
    )
    mla_key = torch.cat(
        [mla_k_nope, mla_k_pe.expand(-1, config.num_attention_heads, -1, -1)], dim=-1
    )

    assert torch.equal(mla_key, orig_key)
    assert torch.equal(mla_val, orig_val)

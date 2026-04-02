# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers import DynamicCache

from tests.utils import failed_ttmlir_compilation
from third_party.tt_forge_models.kimi.kimi_k2.pytorch.loader import ModelLoader

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
)
from .original_modeling_deepseek import DeepseekV3Attention as OrigDeepseekV3Attention
from .utils import MLACache


def test_kimi_k2_single_layer():
    xr.set_device_type("TT")

    loader = ModelLoader(num_layers=1, no_tokenizer=True)
    model = loader.load_model(dtype_override=torch.bfloat16)

    batch_size = 64
    seq_len = 32
    tokens = loader.load_inputs(batch_size=batch_size, seq_len=seq_len)

    compiled_model = torch.compile(model, backend="tt")

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.logits.to("cpu")


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 32. "
    )
)
def test_kimi_k2_rotary_emb_gather_all_rows():
    """Reproduces the TT compiler gather-all-rows bug via DeepseekV3Attention without a KV cache.

    The bug lives in GatherToSliceRepeatConcatConversionPattern in tt-mlir's
    TTIRToTTIRDecomposition pass.  That pattern only fires when startIndices is a
    ttir::ConstantOp.  In the full model, position_ids are generated internally via
    torch.arange which XLA folds into a stablehlo.constant → ttir::ConstantOp.

    When the pattern sees indices [0, 1, ..., N-1] (all rows, no padding):
      starts = 1, ends = 1  →  after decrement: starts = 0, ends = 0
      slicesToConcat = [op.getInput()]  (shape (N, D))
      ConcatOp(output_type=op.getType()=(1, N, D), inputs=[(N, D)], dim=0)
      → verifier error: output dim 0 = 1 but inputs sum to N

    The workaround in tt_forge_models uses rope_seq_len = q_len + 1 so that
    position_ids only index N rows from an (N+1)-row cos table.  With N < N+1 the
    slice is a proper subset and the all-rows pattern is avoided.

    The fix in tt-mlir is to return failure() when starts == 0 && ends == 0 (no
    padding found), letting GatherToEmbeddingConversionPattern handle it instead.
    """
    xr.set_device_type("TT")

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 1

    batch_size = 64
    seq_len = 32

    # Attention without a KV cache: kv_seq_len == q_len, triggering gather-all-rows.
    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16).eval()

    # Wrap the attention so that position_ids are generated *inside* the module via
    # torch.arange.  XLA folds arange(0, seq_len) into a stablehlo.constant which
    # becomes ttir::ConstantOp, enabling GatherToSliceRepeatConcatConversionPattern
    # to match and expose the output-shape bug.  Passing position_ids as an external
    # input leaves them as non-constant placeholders and the pattern never fires.
    class AttentionWithInternalPositionIds(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn

        def forward(self, hidden_states, attention_mask):
            seq = hidden_states.shape[1]
            position_ids = torch.arange(seq, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            return self.attn(hidden_states, attention_mask, position_ids)

    wrapped = AttentionWithInternalPositionIds(attention).eval()

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    # Without a KV cache, kv_seq_len == q_len == seq_len, so the attention mask is square.
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    compiled = torch.compile(wrapped, backend="tt")

    device = torch_xla.device()
    hidden_states = hidden_states.to(device)
    attention_mask = attention_mask.to(device)
    compiled = compiled.to(device)

    with torch.no_grad():
        output = compiled(hidden_states, attention_mask)
        output[0].to("cpu")


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_attention_prefill():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    batch_size = 64
    seq_len = 32
    max_cache_len = 1024
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
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


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_attention_decode():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 1

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
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
    past_key_states = static_cache

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][0]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][1]] = ("_axis_1", None, None, None)

        # Main attention weights, TP across model and batch dimensions
        shard_specs[attention.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.o_proj.weight] = (None, "_axis_0")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "_axis_0")
        return shard_specs

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


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_layer():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config._attn_implementation = "eager"
    config.num_hidden_layers = 1

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer = layer.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
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
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][0]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][1]] = ("_axis_1", None, None, None)

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

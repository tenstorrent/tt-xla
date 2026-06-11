# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, MLACache, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
    ModelLoader,
)


def _attention_shard_spec(attn):
    """Per-attention shard spec, mirrors ``loader.load_shard_spec`` for ``self_attn``."""
    return {
        attn.q_a_proj.weight: (None, "model"),
        attn.q_b_proj.weight: ("model", None),
        attn.kv_a_proj_with_mqa.weight: (None, "model"),
        attn.kv_b_proj.weight: ("model", None),
        attn.o_proj.weight: (None, "model"),
    }


def _moe_shard_spec(mlp_wrapper):
    """A2aSparseMLPWithSharedExperts shard spec, mirrors ``loader.load_shard_spec``."""
    shard_specs = {}
    inner = mlp_wrapper.mlp
    shard_specs[inner.router.gate.weight] = (None, "model")
    shard_specs[inner.experts.gate_proj] = (("model", "batch"), None, None)
    shard_specs[inner.experts.up_proj] = (("model", "batch"), None, None)
    shard_specs[inner.experts.down_proj] = (("model", "batch"), None, None)
    for bias_name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
        b = getattr(inner.experts, bias_name, None)
        if b is not None:
            shard_specs[b] = (("model", "batch"), None)
    shared = mlp_wrapper.shared_experts
    if shared is not None:
        shard_specs[shared.gate_proj.weight] = (None, "model")
        shard_specs[shared.up_proj.weight] = (None, "model")
        shard_specs[shared.down_proj.weight] = ("model", None)
    return shard_specs


def _decoder_layer_shard_spec(layer):
    """Full decoder-layer shard spec (attn + norms + mlp), mirrors ``loader.load_shard_spec``."""
    shard_specs = _attention_shard_spec(layer.self_attn)
    shard_specs[layer.input_layernorm.weight] = ("model",)
    shard_specs[layer.post_attention_layernorm.weight] = ("model",)
    # MoE layers have .mlp wrapped in A2aSparseMLPWithSharedExperts (after
    # enable_sparse_mlp); dense layers keep DeepseekV3MLP with gate/up/down_proj.
    if hasattr(layer.mlp, "shared_experts"):
        shard_specs.update(_moe_shard_spec(layer.mlp))
    else:
        shard_specs[layer.mlp.gate_proj.weight] = ("batch", "model")
        shard_specs[layer.mlp.up_proj.weight] = ("batch", "model")
        shard_specs[layer.mlp.down_proj.weight] = ("model", "batch")
    return shard_specs


def _make_mla_cache(config, batch_size, max_cache_len, dtype=torch.bfloat16):
    """Pre-allocated MLA cache on CPU; mirrors llm_utils.decode_utils.init_mla_cache."""
    cache = MLACache(config=config, max_cache_len=max_cache_len)
    dummy_kv = torch.zeros(
        (batch_size, 1, 1, config.kv_lora_rank), dtype=dtype, device="cpu"
    )
    dummy_pe = torch.zeros(
        (batch_size, 1, 1, config.qk_rope_head_dim), dtype=dtype, device="cpu"
    )
    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)
    return cache


class _HeadWrapper(nn.Module):
    """Composition of the model's final RMSNorm and lm_head — the vocab projection
    chain the benchmark applies after the last decoder layer (see
    ``llm_benchmark.py``: ``model.lm_head(model.model.norm(...))`` via the
    LLMSamplingWrapper)."""

    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_moe_block(batch_size, seq_len):
    """PCC test for the first MoE block of DeepSeek V3.1 with real weights.

    Loads the 4-layer model via the V3.1 ModelLoader (which downloads BF16
    weights and applies enable_sparse_mlp), then isolates the MoE at layer
    ``first_k_dense_replace`` (= 3) and runs it through run_graph_test on a
    Galaxy (4, 8) mesh.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    # first_k_dense_replace=3 -> layer 3 is the first MoE layer. After
    # enable_sparse_mlp the .mlp is A2aSparseMLPWithSharedExperts.
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        shard_specs = _moe_shard_spec(mlp_wrapper)
        # hidden_states: [batch, seq, hidden] — batch on "batch" (4), hidden on "model" (8).
        shard_specs[args[0]] = ("batch", None, "model")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        moe_block,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_attention_prefill(batch_size, seq_len):
    """PCC test for DeepSeek V3.1 attention (prefill) with real weights.

    Loads the 4-layer model, isolates ``self_attn`` from layer 0, and runs a
    prefill-style forward (full causal mask, ``cache_position=arange(seq_len)``)
    against an empty MLA cache. Matches the benchmark setup in
    ``tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers``
    (batch=64, input_sequence_length=128, ``use_mla_cache=True``).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    attention = model.model.layers[0].self_attn
    attention.eval()

    # Benchmark uses max_cache_len == input_sequence_length (see llm_benchmark.py).
    max_cache_len = seq_len

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Additive 4D causal mask: 0 = attend, -inf = mask. Shape (B, 1, q, kv).
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(attn, args, kwargs):
        shard_specs = _attention_shard_spec(attn)
        # hidden_states (B, S, H) — batch on "batch", hidden on "model".
        shard_specs[args[0]] = ("batch", None, "model")
        # attention_mask (B, 1, q, kv) — batch on "batch".
        shard_specs[args[1]] = ("batch", None, None, None)
        # MLA cache: compressed_kv / k_pe (B, 1, max_cache, dim) — batch on "batch".
        cache = args[3]
        shard_specs[cache.layers[0].compressed_kv] = ("batch", None, None, None)
        shard_specs[cache.layers[0].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("prefill_len", [127])
def test_deepseek_v3_1_attention_decode(batch_size, prefill_len):
    """PCC test for DeepSeek V3.1 attention (decode step) with real weights.

    Decodes a single token at ``cache_position=[prefill_len]`` against an
    MLA cache pre-populated by running the same attention on CPU for a
    ``prefill_len``-token prompt. Matches the post-prefill state of the
    benchmark (batch=64, max_cache_len=128 from input_sequence_length=128).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    attention = model.model.layers[0].self_attn
    attention.eval()

    seq_len = 1
    # Pinned to benchmark default (input_sequence_length=128); prefill_len must
    # leave at least one decode slot, so prefill_len < max_cache_len.
    max_cache_len = 128

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    # Warm the cache on CPU so decode sees a realistic post-prefill state. Both
    # the CPU golden run and the TT run will get an independent ``copy.copy``
    # of this populated cache (see torch_device_runner.to_device).
    prefill_hidden = torch.randn(
        (batch_size, prefill_len, config.hidden_size), dtype=torch.bfloat16
    )
    prefill_mask = torch.zeros(
        batch_size, 1, prefill_len, max_cache_len, dtype=torch.bfloat16
    )
    prefill_mask.masked_fill_(
        ~torch.ones(prefill_len, max_cache_len).bool().tril(), float("-inf")
    )
    with torch.no_grad():
        attention(
            prefill_hidden,
            prefill_mask,
            torch.arange(prefill_len).unsqueeze(0),
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.arange(prefill_len, dtype=torch.long),
        )

    # Decode-step inputs: single new token at position ``prefill_len``.
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Decode mask: attend to positions [0, prefill_len], mask out the rest.
    key_idx = torch.arange(max_cache_len)
    mask_row = torch.where(
        key_idx > prefill_len,
        torch.tensor(float("-inf"), dtype=torch.bfloat16),
        torch.tensor(0.0, dtype=torch.bfloat16),
    )
    attention_mask = (
        mask_row.view(1, 1, 1, max_cache_len)
        .expand(batch_size, 1, 1, max_cache_len)
        .contiguous()
    )
    position_ids = torch.tensor([[prefill_len]], dtype=torch.long)
    cache_position = torch.tensor([prefill_len], dtype=torch.long)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(attn, args, kwargs):
        shard_specs = _attention_shard_spec(attn)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[0].compressed_kv] = ("batch", None, None, None)
        shard_specs[cache.layers[0].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "layer_idx",
    [pytest.param(0, id="dense"), pytest.param(3, id="moe")],
)
def test_deepseek_v3_1_decoder_layer_prefill(batch_size, seq_len, layer_idx):
    """PCC test for one full DeepSeek V3.1 decoder layer (prefill).

    Layer 0 has a dense MLP; layer 3 is the first MoE layer
    (``first_k_dense_replace=3``). Exposes the bf16 residual-stream drift and
    the input/post-attention layernorms that the attention/MoE block tests
    don't cover, which is where the per-layer PCC gap likely comes from.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    decoder_layer = model.model.layers[layer_idx]
    decoder_layer.eval()

    max_cache_len = seq_len
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(layer, args, kwargs):
        shard_specs = _decoder_layer_shard_spec(layer)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[layer_idx].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        shard_specs[cache.layers[layer_idx].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        decoder_layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("prefill_len", [127])
@pytest.mark.parametrize(
    "layer_idx",
    [pytest.param(0, id="dense"), pytest.param(3, id="moe")],
)
def test_deepseek_v3_1_decoder_layer_decode(batch_size, prefill_len, layer_idx):
    """PCC test for one full DeepSeek V3.1 decoder layer (decode step).

    Warms layer ``layer_idx``'s MLA cache on CPU via a ``prefill_len``-token
    prefill, then decodes a single token at ``cache_position=[prefill_len]``.
    ``max_cache_len`` is pinned to the benchmark default (128).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    decoder_layer = model.model.layers[layer_idx]
    decoder_layer.eval()

    seq_len = 1
    max_cache_len = 128

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    # Warm the MLA cache on CPU with a real prefill through the layer; both the
    # CPU golden run and the TT run get an independent copy.copy of this
    # populated cache (see torch_device_runner.to_device).
    prefill_hidden = torch.randn(
        (batch_size, prefill_len, config.hidden_size), dtype=torch.bfloat16
    )
    prefill_mask = torch.zeros(
        batch_size, 1, prefill_len, max_cache_len, dtype=torch.bfloat16
    )
    prefill_mask.masked_fill_(
        ~torch.ones(prefill_len, max_cache_len).bool().tril(), float("-inf")
    )
    with torch.no_grad():
        decoder_layer(
            prefill_hidden,
            prefill_mask,
            torch.arange(prefill_len).unsqueeze(0),
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.arange(prefill_len, dtype=torch.long),
        )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    key_idx = torch.arange(max_cache_len)
    mask_row = torch.where(
        key_idx > prefill_len,
        torch.tensor(float("-inf"), dtype=torch.bfloat16),
        torch.tensor(0.0, dtype=torch.bfloat16),
    )
    attention_mask = (
        mask_row.view(1, 1, 1, max_cache_len)
        .expand(batch_size, 1, 1, max_cache_len)
        .contiguous()
    )
    position_ids = torch.tensor([[prefill_len]], dtype=torch.long)
    cache_position = torch.tensor([prefill_len], dtype=torch.long)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(layer, args, kwargs):
        shard_specs = _decoder_layer_shard_spec(layer)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[layer_idx].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        shard_specs[cache.layers[layer_idx].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        decoder_layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_head(batch_size, seq_len):
    """PCC test for the model head (final RMSNorm + lm_head) with real weights.

    Exercises the vocab projection over the residual stream — the part that
    sees the most accumulated bf16 drift in the benchmark and isn't covered
    by any per-block test.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    head = _HeadWrapper(model.model.norm, model.lm_head)
    head.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(head, args, kwargs):
        return {
            args[0]: ("batch", None, "model"),
            head.norm.weight: ("model",),
            # loader.load_shard_spec: lm_head.weight = (None, "model").
            head.lm_head.weight: (None, "model"),
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        head,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


class _GateScores(nn.Module):
    """Returns the dense [tokens, num_experts] routing-score vector the MoE
    gate produces (zeros except at the top-k selected experts). Comparing this
    device-vs-CPU separates the two routed-path failure modes:
      high PCC -> gate selects the SAME experts on device and CPU; the error is
                  downstream in the sparse expert matmul kernels.
      low  PCC -> the device-lowered gate (fp32 logits -> sigmoid -> grouped
                  top-k -> scatter/gather) selects DIFFERENT experts than CPU,
                  e.g. bf16 near-ties flipping under outlier activations.
    """

    def __init__(self, router, num_experts):
        super().__init__()
        from tt_torch.sparse_mlp import _unpack_router_output

        self.router = router
        self.num_experts = num_experts
        self._unpack = _unpack_router_output

    def forward(self, hidden_states):
        out = self.router(hidden_states)
        scores, _indices = self._unpack(out, self.num_experts)  # [tokens, E]
        return scores


class _RoutedOnly(nn.Module):
    """Exposes the ROUTED-only MoE output on both CPU and TT for an
    apples-to-apples comparison.

    - TT: ``A2aSparseMLP.forward`` already returns routed-only (the wrapper adds
      shared separately), so we return it directly.
    - CPU: the A2aSparseMLP CPU golden (``_cpu_forward``) delegates to the
      original ``DeepseekV3MoE`` which computes ``routed + shared`` internally
      (``y = routed; y += shared_experts(x)``). Routed-only golden is therefore
      ``original_moe(x) - shared_experts(x)`` — exact.

    This lets run_graph_test compare the device routed sparse path
    (sparse_matmul / all_to_all dispatch+combine) against a correct routed-only
    reference, with shared-experts removed from both sides.
    """

    def __init__(self, moe_block):
        super().__init__()
        self.mlp = moe_block.mlp  # A2aSparseMLP
        self.shared_experts = moe_block.shared_experts

    def forward(self, hidden_states):
        if hidden_states.device.type == "cpu":
            full, _ = self.mlp._cpu_forward(hidden_states)  # routed + shared
            return full - self.shared_experts(hidden_states)  # routed only
        out, _ = self.mlp(hidden_states)  # TT routed only
        return out


def _capture_real_moe_input(model, loader, moe_block, batch_size, seq_len, diverse=False):
    """Run a CPU prefill and capture the real activation feeding the first MoE
    block (post_attention_layernorm output of layer ``first_k_dense_replace``).

    The existing MoE/decoder-layer tests feed ``torch.randn`` whose
    distribution does not match what 3 real dense layers produce. The full
    benchmark drops from PCC ~0.9999 (3 dense layers) to ~0.978 the moment the
    first MoE layer is added, so the realistic input distribution is needed to
    reproduce the gap in isolation.

    diverse=False (default): tile the benchmark's default prompt across all
    batch rows (matches the benchmark exactly — identical rows route all
    tokens to the same few experts, causing extreme load imbalance).
    diverse=True: random per-row token ids — keeps the real activation
    magnitude/outlier structure but spreads routing evenly. Used to separate
    the routing-skew (capacity-overflow) hypothesis from the magnitude/outlier
    hypothesis.
    """
    if diverse:
        vocab = model.config.vocab_size
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
    elif seq_len is None:
        # Benchmark-faithful: tokenize the default prompt WITHOUT padding, exactly
        # like llm_benchmark.construct_inputs (truncation only, no pad). The
        # prompt is ~16 tokens, so load_inputs(seq_len=128) would pad with ~112
        # identical pad tokens -> artificial routing concentration. This path
        # reproduces the real benchmark prefill input (seq == prompt length).
        if loader.tokenizer is None:
            loader._load_tokenizer()
        prompt = "Here is an exaustive list of the best practices for writing clean code:"
        toks = loader.tokenizer(prompt, return_tensors="pt").input_ids[0]
        input_ids = toks.unsqueeze(0).expand(batch_size, -1).contiguous()
        print(f"[CAPTURE] no-pad benchmark-faithful: seq={input_ids.shape[1]}", flush=True)
    else:
        input_ids = loader.load_inputs(batch_size=batch_size, seq_len=seq_len)
    captured = {}

    def _pre_hook(module, args):
        captured["hidden_states"] = args[0].detach().clone()

    handle = moe_block.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            model.model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    assert "hidden_states" in captured, "MoE block forward_pre_hook never fired"
    return captured["hidden_states"]


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        pytest.param(64, 128, id="bench_scale"),
        pytest.param(4, 32, id="v4_scale"),
    ],
)
@pytest.mark.parametrize(
    "diverse",
    [
        pytest.param(False, id="tiled_prompt"),
        pytest.param(True, id="diverse_rows"),
    ],
)
def test_deepseek_v3_1_moe_block_real_input(batch_size, seq_len, diverse):
    """PCC test for the first MoE block, fed the REAL post-attention-layernorm
    activation from a CPU prefill instead of ``torch.randn``.

    This is the isolated single-layer reproduction of the benchmark PCC gap:
    3 dense layers alone give PCC ~0.9999, but adding this one MoE layer drops
    the full model to ~0.978. ``test_deepseek_v3_1_moe_block`` passes at 0.997
    only because its random input does not match the real activation
    distribution.

    ``tiled_prompt`` matches the benchmark (same prompt across all rows ->
    extreme routing skew). ``diverse_rows`` keeps the real activation
    magnitude but spreads routing -> isolates capacity-overflow vs magnitude.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    # first_k_dense_replace=3 -> layer 3 is the first MoE layer.
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    # DEBUG knob: swap sparse_matmul expert compute for dense torch.bmm
    # (dispatch/combine unchanged) to localize the magnitude effect.
    import os

    if os.environ.get("DSV3_DENSE_MATMUL") == "1":
        moe_block.mlp.use_dense_matmul = True

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len, diverse=diverse
    )
    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        shard_specs = _moe_shard_spec(mlp_wrapper)
        shard_specs[args[0]] = ("batch", None, "model")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        moe_block,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [16, 32, 128])
def test_deepseek_v3_1_shared_experts_real_input(batch_size, seq_len):
    """PCC for ONLY the shared-experts dense MLP of layer-3, fed the same real
    activation as the MoE block.

    The MoE block output = routed_experts + shared_experts. The shared experts
    are a plain dense DeepseekV3MLP (same op family as the first 3 dense layers,
    which give PCC ~0.9999). If this passes at ~0.99 while the full MoE block is
    ~0.55, the error is isolated to the ROUTED path (A2aSparseMLP:
    sparse_matmul / all_to_all_dispatch / combine), not dense matmul.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()
    shared = moe_block.shared_experts
    assert shared is not None, "layer-3 MoE has no shared_experts"
    shared.eval()

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len
    )
    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(shared_mlp, args, kwargs):
        # Mirrors the shared-experts entries in _moe_shard_spec.
        return {
            args[0]: ("batch", None, "model"),
            shared_mlp.gate_proj.weight: (None, "model"),
            shared_mlp.up_proj.weight: (None, "model"),
            shared_mlp.down_proj.weight: ("model", None),
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        shared,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_routed_only_real_input(batch_size, seq_len):
    """PCC for the ROUTED-only MoE path (shared experts removed from both sides),
    fed the real layer-3 activation.

    Final decomposition: shared-experts (dense) already passes at ~0.997 and the
    full MoE block is ~0.55. This compares the device routed sparse path
    (sparse_matmul + all_to_all dispatch/combine) against
    ``original_moe(x) - shared(x)``.
      ~0.55  -> the routed sparse kernels are the numerics culprit (tt-metal).
      ~0.99  -> routed math is fine; the wrapper's shared add/scaling is at fault.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len
    )
    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)

    routed_only = _RoutedOnly(moe_block)
    routed_only.eval()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mod, args, kwargs):
        # Reuse the MoE shard spec (routed experts + shared weights) keyed off
        # the wrapped A2aSparseMLPWithSharedExperts-like layout.
        shard_specs = _moe_shard_spec(mod)
        shard_specs[args[0]] = ("batch", None, "model")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        routed_only,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_gate_scores_real_input(batch_size, seq_len):
    """PCC for ONLY the MoE gate's dense routing scores [tokens, num_experts],
    fed the real layer-3 activation.

    Separates the two routed-path failure modes (see _GateScores):
      high PCC -> device & CPU pick the same experts; sparse expert matmul is
                  the numerics culprit.
      low  PCC -> device gate selects different experts (routing divergence)
                  under the outlier activation distribution.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len
    )

    gate_scores = _GateScores(moe_block.mlp.router, config.n_routed_experts)
    gate_scores.eval()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mod, args, kwargs):
        return {
            args[0]: ("batch", None, "model"),
            mod.router.gate.weight: (None, "model"),
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        gate_scores,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
def test_deepseek_v3_1_moe_block_nopad(batch_size):
    """Benchmark-FAITHFUL MoE block PCC: real layer-3 activation captured with
    the prompt tokenized WITHOUT padding (seq == prompt length ~16), exactly as
    llm_benchmark feeds it. The seq=128 captures pad ~112 identical pad tokens
    that artificially concentrate routing; this isolates whether the MoE-block
    PCC gap is real at benchmark conditions or a padding artifact.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len=None
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        shard_specs = _moe_shard_spec(mlp_wrapper)
        shard_specs[args[0]] = ("batch", None, "model")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        moe_block,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
def test_deepseek_v3_1_gate_scores_nopad(batch_size):
    """Benchmark-FAITHFUL gate-scores PCC (no padding). Pairs with
    test_deepseek_v3_1_moe_block_nopad to check whether the gate selection
    divergence persists at real benchmark conditions (seq ~16, no pad tokens)."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len=None
    )

    gate_scores = _GateScores(moe_block.mlp.router, config.n_routed_experts)
    gate_scores.eval()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mod, args, kwargs):
        return {
            args[0]: ("batch", None, "model"),
            mod.router.gate.weight: (None, "model"),
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        gate_scores,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


class _FrozenRouter(nn.Module):
    """Returns a FIXED (scores, indices) captured once on CPU, ignoring input.

    Swapped in for A2aSparseMLP.router so the DEVICE routed path uses the exact
    golden (CPU-computed) expert routing. The CPU golden path (_cpu_forward ->
    original DeepseekV3MoE) recomputes the same routing from the same input, so
    device and CPU end up with identical routing. This removes gate-selection
    divergence and isolates the sparse expert COMPUTE
    (sparse_matmul / all_to_all dispatch+combine).
    """

    def __init__(self, real_router, sample_hidden):
        super().__init__()
        with torch.no_grad():
            scores, indices = real_router(sample_hidden)
        self.register_buffer("scores", scores.detach().clone())
        self.register_buffer("indices", indices.detach().clone())
        self.gate = real_router.gate  # keep ref so A2aSparseMLP passes 3D input

    def forward(self, hidden_states, *args, **kwargs):
        return self.scores, self.indices


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize(
    "seq_len",
    [
        pytest.param(None, id="nopad"),
        pytest.param(32, id="seq32"),
        pytest.param(128, id="bench_scale"),
    ],
)
@pytest.mark.parametrize(
    "diverse",
    [
        pytest.param(False, id="tiled_prompt"),
        pytest.param(True, id="diverse_rows"),
    ],
)
def test_deepseek_v3_1_moe_block_frozen_routing_nopad(batch_size, seq_len, diverse):
    """MoE block with FROZEN (golden) routing.

    Isolates the sparse expert COMPUTE from gate-selection divergence:
      ~0.99 -> compute is correct; the gate selection is the whole problem.
      ~0.35 -> sparse_matmul / dispatch / combine kernels are broken regardless
               of routing (the dominant culprit at real conditions).

    nopad (seq=16, sub-tile) is a degenerate case; bench_scale (seq=128, padded
    like the real benchmark) is the faithful conditions where the full model
    sits at PCC ~0.978.

    tiled_prompt = identical rows (extreme routing concentration, matches the
    benchmark); diverse_rows = spread routing. Comparing the two under FROZEN
    (identical device/CPU) routing isolates whether the COMPUTE path loses
    accuracy specifically under routing concentration (dispatch capacity
    overflow / token drops).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    capture_seq = seq_len if seq_len is not None else 128
    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size,
        seq_len=(capture_seq if diverse else seq_len), diverse=diverse,
    )

    # Freeze the device router to the golden CPU routing for this exact input.
    frozen = _FrozenRouter(moe_block.mlp.router, hidden_states)
    moe_block.mlp.router = frozen

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        shard_specs = _moe_shard_spec(mlp_wrapper)
        shard_specs[args[0]] = ("batch", None, "model")
        # Frozen routing tensors are [B*S, E] / [B*S, K]; shard the token dim by
        # batch to match the hidden-state batch sharding.
        shard_specs[mlp_wrapper.mlp.router.scores] = ("batch", None)
        shard_specs[mlp_wrapper.mlp.router.indices] = ("batch", None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    # [DEBUG] numerics knobs to attack the per-layer residual (~0.964 at seq=32):
    #   FP32_DEST_ACC=1 -> fp32 destination accumulation for matmuls
    #   MATH_FIDELITY=hifi4|... -> matmul math fidelity
    import os as _os
    from tests.infra.testers.compiler_config import CompilerConfig as _CC

    _cc = None
    if (
        _os.environ.get("FP32_DEST_ACC")
        or _os.environ.get("MATH_FIDELITY")
        or _os.environ.get("OPT_LEVEL")
        or _os.environ.get("EXPORT_PATH")
    ):
        _cc = _CC(
            fp32_dest_acc_en=(_os.environ.get("FP32_DEST_ACC") == "1") or None,
            math_fidelity=_os.environ.get("MATH_FIDELITY") or None,
            optimization_level=int(_os.environ.get("OPT_LEVEL", "0")),
            export_path=_os.environ.get("EXPORT_PATH", ""),
            export_model_name=_os.environ.get("EXPORT_NAME", "dsv31_moe"),
        )

    run_graph_test(
        moe_block,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=_cc,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize(
    "diverse",
    [
        pytest.param(False, id="tiled_prompt"),
        pytest.param(True, id="diverse_rows"),
    ],
)
def test_deepseek_v3_1_routed_only_frozen_routing(batch_size, diverse):
    """ROUTED-only path (shared removed) with FROZEN (golden) routing, bench scale.

    The single cleanest probe of the sparse routed KERNELS:
      - shared experts removed -> no large-magnitude masking of the routed error
      - routing frozen to CPU golden -> device & CPU use IDENTICAL expert
        selection, so any PCC drop is purely the on-device routed compute
        (all_to_all_dispatch / sparse_matmul / all_to_all_combine).

    ~0.99 -> routed kernels are correct; the benchmark gap is gate selection.
    <0.9  -> the routed kernels themselves are the numerics culprit (tt-metal),
             independent of routing.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()

    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    import os as _os
    _rseq = int(_os.environ.get("ROUTED_SEQ", "128"))
    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len=_rseq, diverse=diverse
    )

    # Freeze the device router to the golden CPU routing for this exact input.
    frozen = _FrozenRouter(moe_block.mlp.router, hidden_states)
    moe_block.mlp.router = frozen

    routed_only = _RoutedOnly(moe_block)
    routed_only.eval()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    import os as _os
    _repl = _os.environ.get("REPL_BATCH") == "1"

    def get_shard_spec(mod, args, kwargs):
        shard_specs = _moe_shard_spec(mod)
        if _repl:
            # Replicate the token/batch dim across the cluster axis instead of
            # sharding it there. The all_to_all_dispatch kernel assumes the batch
            # is replicated along the cluster axis; sharding it there is suspected
            # to drop half the tokens.
            shard_specs[args[0]] = (None, None, "model")
            shard_specs[mod.mlp.router.scores] = (None, None)
            shard_specs[mod.mlp.router.indices] = (None, None)
        else:
            shard_specs[args[0]] = ("batch", None, "model")
            shard_specs[mod.mlp.router.scores] = ("batch", None)
            shard_specs[mod.mlp.router.indices] = ("batch", None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        routed_only,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


class _RoutedSharedSplit(nn.Module):
    """Returns stack([routed, shared]) so a custom comparator can measure each
    part separately and how they combine (to explain why full block PCC can be
    LOWER than both routed-only and shared-only)."""

    def __init__(self, moe_block):
        super().__init__()
        self.mlp = moe_block.mlp  # A2aSparseMLP
        self.shared_experts = moe_block.shared_experts

    def forward(self, hidden_states):
        shared = self.shared_experts(hidden_states)
        if hidden_states.device.type == "cpu":
            full, _ = self.mlp._cpu_forward(hidden_states)  # routed + shared
            routed = full - shared
        else:
            routed, _ = self.mlp(hidden_states)  # device routed only
        return torch.stack([routed, shared], dim=0)  # [2, B, S, H]


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [None, 32, 128])
def test_deepseek_v3_1_routed_shared_split(batch_size, seq_len):
    """Measure routed vs shared separately (frozen routing) + how they combine,
    to explain full-block PCC < both parts (e.g. seq=16 block 0.36 vs routed
    0.83, shared 0.997)."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()
    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size, seq_len=seq_len
    )
    frozen = _FrozenRouter(moe_block.mlp.router, hidden_states)
    moe_block.mlp.router = frozen
    split = _RoutedSharedSplit(moe_block)
    split.eval()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, axis_names)

    def get_shard_spec(mod, args, kwargs):
        ss = _moe_shard_spec(mod)
        ss[args[0]] = ("batch", None, "model")
        ss[mod.mlp.router.scores] = ("batch", None)
        ss[mod.mlp.router.indices] = ("batch", None)
        return ss

    def _pcc(a, b):
        a = a.detach().to(torch.float32).flatten()
        b = b.detach().to(torch.float32).flatten()
        a = a - a.mean(); b = b - b.mean()
        d = a.norm() * b.norm()
        return (torch.dot(a, b) / d).item() if d > 0 else float("nan")

    def comparator(tt_res, cpu_res, args, kwargs):
        tt = tt_res.cpu().to(torch.float32); cp = cpu_res.cpu().to(torch.float32)
        import os as _os2
        if _os2.environ.get("SPLIT_SAVE"):
            torch.save({"tt": tt, "cp": cp}, _os2.environ["SPLIT_SAVE"])
        rd_t, sh_t = tt[0], tt[1]
        rd_c, sh_c = cp[0], cp[1]
        blk_t, blk_c = rd_t + sh_t, rd_c + sh_c
        cos_rs = _pcc(rd_c, sh_c)  # do routed & shared signals cancel? (neg = cancel)
        print(
            f"\n[SPLIT seq={seq_len}] "
            f"PCC routed={_pcc(rd_t, rd_c):.4f} shared={_pcc(sh_t, sh_c):.4f} "
            f"block={_pcc(blk_t, blk_c):.4f}\n"
            f"  |routed|cpu={rd_c.abs().mean():.4f} |shared|cpu={sh_c.abs().mean():.4f} "
            f"|block|cpu={blk_c.abs().mean():.4f}  corr(routed,shared)={cos_rs:.4f}\n"
            f"  routed absmax c={rd_c.abs().max():.3f} t={rd_t.abs().max():.3f}; "
            f"err |routed_t-routed_c|mean={(rd_t-rd_c).abs().mean():.4f} "
            f"|shared_t-shared_c|mean={(sh_t-sh_c).abs().mean():.4f}",
            flush=True,
        )

    run_graph_test(
        split,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.0)),
        custom_comparator=comparator,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
def test_dump_sparse_matmul_io(batch_size):
    """Register a runtime post-op hook that dumps every ttnn.sparse_matmul's
    input+output (shard 0) to tmp/op_dump, then runs the real-shape MoE block.
    Offline we recompute the golden from the dumped inputs and PCC vs the dumped
    output to pin the sparse_matmul hardware kernel. (chisel's own
    sparse_matmul_golden is undefined, so we bypass it.)"""
    import json
    import os as _os
    from _ttmlir_runtime import runtime as ttrt
    from chisel.utils import get_torch_tensor

    DUMP = _os.environ.get("OP_DUMP_DIR", "/localdev/sshon/tt-xla/tmp/op_dump")
    _diverse = _os.environ.get("DUMP_DIVERSE") == "1"
    _os.makedirs(DUMP, exist_ok=True)

    counters = {}
    records = []

    def shards(prog_ctx, ref):
        t = ttrt.retrieve_tensor_from_pool(prog_ctx, ref)
        hs = ttrt.to_host(t, untilize=True)
        return [get_torch_tensor(s) for s in hs]

    _all_ops_log = _os.environ.get("ALL_OPS_LOG")

    def post_op(binary, prog_ctx, op_ctx):
        try:
            asm = ttrt.get_op_debug_str(op_ctx)
        except Exception:
            return
        # ALL_OPS_LOG: append EVERY op's first line (op name + shapes) to a file
        # to reveal the full TTNN op sequence (reshapes / collectives /
        # reduce_scatter / all_gather) assembled around the sparse path.
        if _all_ops_log:
            try:
                with open(_all_ops_log, "a") as _f:
                    _f.write(asm.strip().split("\n")[0][:400] + "\n")
            except Exception:
                pass
        tag = None
        for k in ("all_to_all_dispatch", "all_to_all_combine",
                  "moe_expert_token_remap", "sparse_matmul",
                  "reduce_scatter", "all_gather"):
            if k in asm:
                tag = {"all_to_all_dispatch": "disp", "all_to_all_combine": "comb",
                       "moe_expert_token_remap": "remap", "sparse_matmul": "sm",
                       "reduce_scatter": "rs", "all_gather": "ag"}[k]
                break
        if tag is None:
            return
        i = counters.get(tag, 0)
        counters[tag] = i + 1
        try:
            ins = ttrt.get_op_input_refs(op_ctx)
            outs = ttrt.get_op_output_refs(op_ctx)
            rec = {"tag": tag, "idx": i, "asm": asm[:300]}
            # For dispatch, save ALL shards (to decode the mesh layout); others shard0.
            allshards = tag in ("disp", "comb", "rs", "ag", "sm")
            # METADATA_ONLY: only save the small int metadata (dispatch out1 =
            # gathered expert indices; combine in1 = metadata) to cheaply probe
            # token survival across batch sizes without the multi-GB hidden/H tensors.
            meta_only = _os.environ.get("DUMP_METADATA_ONLY") == "1"
            for j, r in enumerate(ins):
                if meta_only and not (tag == "comb" and j == 1):
                    continue
                sh = shards(prog_ctx, r)
                if allshards:
                    torch.save(torch.stack(sh), f"{DUMP}/{tag}{i}_in{j}_all.pt")
                torch.save(sh[0], f"{DUMP}/{tag}{i}_in{j}.pt")
                rec[f"in{j}"] = [tuple(sh[0].shape), str(sh[0].dtype), len(sh)]
            for j, r in enumerate(outs):
                if meta_only and not (tag == "disp" and j == 1):
                    continue
                sh = shards(prog_ctx, r)
                if allshards:
                    torch.save(torch.stack(sh), f"{DUMP}/{tag}{i}_out{j}_all.pt")
                torch.save(sh[0], f"{DUMP}/{tag}{i}_out{j}.pt")
                rec[f"out{j}"] = [tuple(sh[0].shape), str(sh[0].dtype), len(sh)]
            records.append(rec)
        except Exception as e:
            records.append({"idx": i, "error": repr(e)[:200]})

    batch_size = int(_os.environ.get("DUMP_BATCH", batch_size))
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()
    _dump_seq = int(_os.environ.get("DUMP_SEQ", "16"))
    hidden_states = _capture_real_moe_input(
        model, loader, moe_block, batch_size,
        seq_len=(_dump_seq if _diverse else None), diverse=_diverse,
    )

    if _os.environ.get("DUMP_FROZEN") == "1":
        frozen = _FrozenRouter(moe_block.mlp.router, hidden_states)
        moe_block.mlp.router = frozen

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        ss = _moe_shard_spec(mlp_wrapper)
        ss[args[0]] = ("batch", None, "model")
        if _os.environ.get("DUMP_FROZEN") == "1":
            ss[mlp_wrapper.mlp.router.scores] = ("batch", None)
            ss[mlp_wrapper.mlp.router.indices] = ("batch", None)
        return ss

    ttrt.DebugHooks.get(post_op=post_op)
    try:
        try:
            run_graph_test(
                moe_block, [hidden_states], framework=Framework.TORCH, mesh=mesh,
                shard_spec_fn=get_shard_spec,
                comparison_config=ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.0)),
            )
        except Exception:
            pass
    finally:
        ttrt.unregister_hooks()

    with open(f"{DUMP}/records.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"[DUMP] {len(records)} sparse_matmul ops dumped to {DUMP}")
    for r in records:
        print(r)


class _GateSelection(nn.Module):
    """Returns a BINARY [tokens, num_experts] multi-hot of the top-k selected
    experts (1.0 at each selected expert, weights ignored). Comparing device vs
    CPU PCC measures pure expert-SELECTION overlap: 1.0 = identical experts
    chosen; lower = the device gate's topk picks different experts than CPU
    (invisible to every downstream per-op kernel check, since they all run on
    whatever routing the device produced)."""

    def __init__(self, router, num_experts):
        super().__init__()
        from tt_torch.sparse_mlp import _unpack_router_output

        self.router = router
        self.num_experts = num_experts
        self._unpack = _unpack_router_output

    def forward(self, hidden_states):
        out = self.router(hidden_states)
        scores, indices = self._unpack(out, self.num_experts)  # indices [tokens, K]
        rng = torch.arange(self.num_experts, device=indices.device)
        onehot = (indices.unsqueeze(-1) == rng).to(scores.dtype)  # [tokens, K, E]
        return onehot.sum(dim=1)  # [tokens, E] multi-hot of selected experts


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
def test_deepseek_v3_1_gate_selection_nopad(batch_size):
    """Pure expert-SELECTION overlap (device vs CPU) at real benchmark shape."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()
    hidden_states = _capture_real_moe_input(model, loader, moe_block, batch_size, seq_len=None)
    sel = _GateSelection(moe_block.mlp.router, config.n_routed_experts)
    sel.eval()
    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, axis_names)

    def get_shard_spec(mod, args, kwargs):
        return {args[0]: ("batch", None, "model"), mod.router.gate.weight: (None, "model")}

    run_graph_test(
        sel, [hidden_states], framework=Framework.TORCH, mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.999)),
    )


class _GateStage(nn.Module):
    """Returns an intermediate of the MoEGate to localize where device diverges
    from CPU: 'logits' (fp32 matmul), 'scores' (sigmoid), 'group_scores'
    (continuous, pre-final-topk). Continuous stages with PCC~1.0 mean the
    divergence is purely the discrete top-k on near-tie scores."""

    def __init__(self, gate, stage):
        super().__init__()
        self.gate = gate  # raw MoEGate
        self.stage = stage

    def forward(self, hidden_states):
        import torch.nn.functional as F

        g = self.gate
        h = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(h.to(torch.float32), g.weight.to(torch.float32), None)
        if self.stage == "logits":
            return logits
        scores = logits.sigmoid()
        if self.stage == "scores":
            return scores
        n = scores.shape[0]
        sfc = scores + g.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            sfc.view(n, g.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, n_group]
        if self.stage == "group_scores":
            return group_scores
        # group selection: top-`topk_group` groups, as a [n, n_group] multi-hot
        group_idx = torch.topk(group_scores, k=g.topk_group, dim=-1, sorted=False)[1]
        rng = torch.arange(g.n_group, device=group_idx.device)
        return (group_idx.unsqueeze(-1) == rng).to(scores.dtype).sum(dim=1)  # [n, n_group]


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("stage", ["logits", "scores", "group_scores", "group_sel"])
def test_deepseek_v3_1_gate_stage_nopad(batch_size, stage):
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config
    model.eval()
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()
    hidden_states = _capture_real_moe_input(model, loader, moe_block, batch_size, seq_len=None)
    raw_gate = getattr(moe_block.mlp.router, "gate", moe_block.mlp.router)
    mod = _GateStage(raw_gate, stage)
    mod.eval()
    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, axis_names)

    def get_shard_spec(m, args, kwargs):
        return {args[0]: ("batch", None, "model"), m.gate.weight: (None, "model")}

    run_graph_test(
        mod, [hidden_states], framework=Framework.TORCH, mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.9999)),
    )

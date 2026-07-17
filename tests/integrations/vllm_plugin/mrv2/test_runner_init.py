# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner __init__ state construction.

``TTModelRunnerV2.__init__`` (see vllm_tt/model_runner_v2.py) extracts config
scalars and builds the split v2 state. It reads a fixed set of vllm_config
attributes, so a duck-typed fake config exercises it on cpu with no engine, no
model, and no TT hardware (device=cpu). ``load_model`` needs a real model/loader
and is validated at engine stand-up, not here.

They pin the wiring: scalar/SMEM-cap derivation, the token-padding ladder, the
constructed state tables, the not-yet-loaded (None) model handles, and the
single-device guard.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates
from vllm_tt.vllm_distributed_utils import ParallelismMode


@contextmanager
def fake_mesh(num_devices):
    """Stub the torch_xla runtime/SPMD calls so mesh build is hw-independent."""
    with patch(
        "torch_xla.runtime.global_runtime_device_count", return_value=num_devices
    ), patch(
        "torch_xla.distributed.spmd.Mesh", side_effect=lambda ids, shape, names: shape
    ), patch(
        "torch_xla.distributed.spmd.set_global_mesh"
    ):
        yield


def make_vllm_config(**tt_overrides):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=2),
        dtype=torch.bfloat16,
        max_model_len=256,
        get_sliding_window=lambda: None,
        get_num_layers_by_block_type=lambda pc, t: 2,
        get_num_attention_heads=lambda pc: 8,
        get_num_kv_heads=lambda pc: 8,
        get_head_size=lambda: 64,
        get_vocab_size=lambda: 1000,
        get_inputs_embeds_size=lambda: 512,
        is_multimodal_model=False,
    )
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
    scheduler_config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048)
    # additional_config is a plain dict (the runner builds TTConfig(**it), as the
    # engine does). min_context_len pins the token-padding ladder deterministically.
    additional_config = {"min_context_len": 32}
    additional_config.update(tt_overrides)
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=object(),
        load_config=object(),
        lora_config=None,
        additional_config=additional_config,
    )


@pytest.mark.push
@pytest.mark.cpu
def test_init_scalars_and_smem_caps():
    r = TTModelRunnerV2(make_vllm_config(), torch.device("cpu"))

    assert r.block_size == 32
    assert r.max_model_len == 256
    assert r.max_num_reqs == 8
    assert r.vocab_size == 1000
    assert r.max_num_blocks_per_req == 8  # cdiv(256, 32)
    assert r.num_kv_heads == 8
    assert r.head_size == 64
    assert r.kv_cache_dtype == torch.bfloat16  # cache_dtype="auto" -> model dtype
    assert r.supports_mm_inputs is False
    # No VLLM_TPU_MOST_MODEL_LEN by default -> only the max-model-len cap applies.
    assert r.num_reqs_most_model_len is None
    # get_max_num_seqs(256, 32) is huge, so the cap is max_num_reqs.
    assert r.num_reqs_max_model_len == 8
    # These fall back to max_num_reqs when the TTConfig fields are unset (None).
    assert r.min_num_reqs == (
        r.max_num_reqs if r.tt_config.min_num_seqs is None else r.tt_config.min_num_seqs
    )
    assert r.max_prefill_num_reqs == (
        r.max_num_reqs
        if r.tt_config.max_prefill_num_seqs is None
        else r.tt_config.max_prefill_num_seqs
    )


@pytest.mark.push
@pytest.mark.cpu
def test_init_token_padding_ladder():
    r = TTModelRunnerV2(make_vllm_config(), torch.device("cpu"))
    # _get_token_paddings(32, min(2048, 256)=256): [1] + powers of two up to >=256.
    assert r.num_tokens_paddings == [1, 32, 64, 128, 256]
    assert r.max_num_tokens == 256


@pytest.mark.push
@pytest.mark.cpu
def test_init_decode_only_restricts_paddings():
    r = TTModelRunnerV2(make_vllm_config(decode_only=True), torch.device("cpu"))
    assert r.num_tokens_paddings == [1]
    assert r.max_num_tokens == 1


@pytest.mark.push
@pytest.mark.cpu
def test_init_builds_split_state_tables():
    r = TTModelRunnerV2(make_vllm_config(), torch.device("cpu"))

    assert isinstance(r.req_states, TTRequestState)
    assert r.req_states.max_num_reqs == 8
    assert r.req_states.max_model_len == 256
    assert isinstance(r.sampling_states, TTSamplingStates)
    # Block table + input buffers are constructed and correctly sized.
    assert hasattr(r.block_table, "add_row")
    assert tuple(r.input_buffers.input_ids.shape) == (r.max_num_tokens,)
    # Runtime-side state initialised empty.
    assert r.encoder_cache == {}
    assert r.num_prompt_logprobs == {}
    assert r.kv_caches == []
    assert r.scheduler_output is None
    assert r.dp_size == 1
    assert r.enable_tensor_parallel is False


@pytest.mark.push
@pytest.mark.cpu
def test_init_model_handles_none_until_load():
    r = TTModelRunnerV2(make_vllm_config(), torch.device("cpu"))
    assert r.model is None
    assert r.model_state is None
    assert r.sampler is None
    assert r.attention_layer_names == ()


@pytest.mark.push
@pytest.mark.cpu
def test_init_single_device_when_flags_off():
    r = TTModelRunnerV2(make_vllm_config(), torch.device("cpu"))
    assert r.parallel_mode == ParallelismMode.DISABLED
    assert r.mesh is None
    assert r.dp_size == 1


@pytest.mark.push
@pytest.mark.cpu
def test_init_tp_only_builds_2d_mesh():
    with fake_mesh(8):
        r = TTModelRunnerV2(
            make_vllm_config(enable_tensor_parallel=True), torch.device("cpu")
        )
    # Pure TP on 8 devices -> (2,4) mesh, but dp_size stays 1 (batch axis is a TP axis).
    assert r.parallel_mode == ParallelismMode.TENSOR_PARALLEL_ONLY_2D
    assert r.mesh == (2, 4)
    assert r.dp_size == 1


@pytest.mark.push
@pytest.mark.cpu
def test_init_dp_tp_sets_dp_size():
    with fake_mesh(8):
        r = TTModelRunnerV2(
            make_vllm_config(enable_tensor_parallel=True, enable_data_parallel=True),
            torch.device("cpu"),
        )
    assert r.parallel_mode == ParallelismMode.DATA_TENSOR_PARALLEL
    assert r.dp_size == 2  # mesh_shape[0] of (2,4)
    assert r.max_num_reqs % r.dp_size == 0


@pytest.mark.push
@pytest.mark.cpu
def test_init_single_device_disables_parallel():
    with fake_mesh(1):
        r = TTModelRunnerV2(
            make_vllm_config(enable_tensor_parallel=True, enable_data_parallel=True),
            torch.device("cpu"),
        )
    # One device -> both disabled, falls back to the single-device path.
    assert r.parallel_mode == ParallelismMode.DISABLED
    assert r.mesh is None
    assert r.dp_size == 1


@pytest.mark.push
@pytest.mark.cpu
def test_init_no_layer_override_by_default():
    cfg = make_vllm_config()
    r = TTModelRunnerV2(cfg, torch.device("cpu"))
    # Default num_hidden_layers=0 -> no override, hf_config untouched.
    assert r._original_num_layers is None
    assert r._target_num_layers is None
    assert cfg.model_config.hf_config.num_hidden_layers == 2


@pytest.mark.push
@pytest.mark.cpu
def test_init_applies_layer_override():
    cfg = make_vllm_config(num_hidden_layers=1)
    r = TTModelRunnerV2(cfg, torch.device("cpu"))
    # Override mutates hf_config so only the target layers get built.
    assert r._original_num_layers == 2
    assert r._target_num_layers == 1
    assert cfg.model_config.hf_config.num_hidden_layers == 1
    # The weight filter drops layers >= target and keeps the rest.
    weights = [
        ("model.layers.0.self_attn.qkv_proj.weight", 0),
        ("model.layers.1.self_attn.qkv_proj.weight", 1),
        ("model.embed_tokens.weight", 2),
    ]
    kept = [n for n, _ in r._filter_weights_for_layer_override(iter(weights))]
    assert kept == [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.embed_tokens.weight",
    ]

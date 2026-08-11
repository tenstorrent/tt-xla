# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 weight_dtype_overrides application in load_model.

Tests verify that weight_dtype_overrides is applied when configured, skipped when
None, and takes precedence over experimental_weight_dtype (for host-side observable
side effects like parametrization count).
"""

from types import SimpleNamespace

import pytest
import torch
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from vllm_tt.platform import TTConfig


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
        uses_mrope=False,
    )
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
    scheduler_config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048)
    additional_config = {"min_context_len": 32}
    additional_config.update(tt_overrides)
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=object(),
        load_config=object(),
        lora_config=None,
        speculative_config=None,
        additional_config=additional_config,
    )


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_config_when_set():
    cfg = make_vllm_config(
        weight_dtype_overrides={"0.weight": "bfp_bf8", "1.weight": "bfp_bf4"}
    )
    tt_config = TTConfig(**cfg.additional_config)

    assert tt_config.weight_dtype_overrides is not None
    assert tt_config.weight_dtype_overrides == {
        "0.weight": "bfp_bf8",
        "1.weight": "bfp_bf4",
    }


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_config_none_by_default():
    cfg = make_vllm_config()
    tt_config = TTConfig(**cfg.additional_config)

    assert tt_config.weight_dtype_overrides is None


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_config_precedence_over_experimental():
    cfg = make_vllm_config(
        weight_dtype_overrides={"0.weight": "bfp_bf4"},
        experimental_weight_dtype="bfp_bf8",
    )
    tt_config = TTConfig(**cfg.additional_config)

    assert tt_config.weight_dtype_overrides == {"0.weight": "bfp_bf4"}
    assert tt_config.experimental_weight_dtype == "bfp_bf8"


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_apply_with_simple_model():
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
    )

    applied = apply_weight_dtype_overrides(
        model, {"0.weight": "bfp_bf8", "1.weight": "bfp_bf4"}
    )

    assert len(applied) == 2
    assert ("0.weight", "bfp_bf8") in applied
    assert ("1.weight", "bfp_bf4") in applied


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_apply_with_default_key():
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
    )

    applied = apply_weight_dtype_overrides(
        model, {"0.weight": "bfp_bf4", "default": "bfp_bf8"}
    )

    assert len(applied) == 2
    dtypes = {param: dtype for param, dtype in applied}
    assert dtypes["0.weight"] == "bfp_bf4"
    assert dtypes["1.weight"] == "bfp_bf8"


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_apply_with_glob_pattern():
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
    )

    applied = apply_weight_dtype_overrides(model, {"*.weight": "bfp_bf8"})

    assert len(applied) == 2
    for param, dtype in applied:
        assert dtype == "bfp_bf8"


@pytest.mark.push
@pytest.mark.cpu
def test_weight_dtype_overrides_apply_str_config():
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
    )

    applied = apply_weight_dtype_overrides(model, "bfp_bf4")

    assert len(applied) == 2
    for param, dtype in applied:
        assert dtype == "bfp_bf4"

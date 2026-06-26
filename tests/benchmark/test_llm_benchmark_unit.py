# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the pure logic in benchmarks.llm_benchmark.

These do not require Tenstorrent hardware: they exercise the env/mode parsing,
perf summarization, compile-option assembly, and PCC/accuracy scoring in
isolation so the benchmark's behavior-critical bookkeeping has fast regression
coverage independent of the on-device integration path.
"""

import pytest
import torch

from benchmarks.llm_benchmark import (
    CompileConfig,
    PccMode,
    build_compile_options,
    evaluate_pcc,
    summarize_perf,
)


# --------------------------------------------------------------------------- #
# PccMode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, pcc_only, assert_prefill, assert_decode, isolated",
    [
        (None, False, True, True, False),
        ("", False, True, True, False),
        ("prefill", True, True, False, False),
        ("decode", True, False, True, True),
        ("both", True, True, True, True),
        ("  BoTh  ", True, True, True, True),  # trimmed + lowercased
        ("garbage", False, True, True, False),  # unknown -> full run
    ],
)
def test_pcc_mode_from_string(value, pcc_only, assert_prefill, assert_decode, isolated):
    mode = PccMode.from_string(value)
    assert mode.pcc_only is pcc_only
    assert mode.assert_prefill is assert_prefill
    assert mode.assert_decode is assert_decode
    assert mode.isolated is isolated


def test_pcc_mode_from_env_delegates(monkeypatch):
    monkeypatch.setenv("TT_PCC_MODE", "decode")
    assert PccMode.from_env() == PccMode.from_string("decode")
    monkeypatch.delenv("TT_PCC_MODE", raising=False)
    assert PccMode.from_env() == PccMode.from_string("")


# --------------------------------------------------------------------------- #
# summarize_perf
# --------------------------------------------------------------------------- #
def test_summarize_perf_basic():
    # 1 prefill (100ms) + 2 decode (50ms each), times in nanoseconds.
    times = [100_000_000, 50_000_000, 50_000_000]
    perf = summarize_perf(times, decode_only=False)
    assert perf.ttft_ms == pytest.approx(100.0)
    assert perf.decode_total_time == pytest.approx(0.1)
    assert perf.decode_total_tokens == 2
    assert perf.tokens_per_second == pytest.approx(20.0)


def test_summarize_perf_decode_only_has_zero_ttft():
    times = [50_000_000, 50_000_000, 50_000_000]
    perf = summarize_perf(times, decode_only=True)
    assert perf.ttft_ms == 0.0
    # decode metrics still come from times[1:]
    assert perf.decode_total_tokens == 2
    assert perf.tokens_per_second == pytest.approx(20.0)


def test_summarize_perf_empty():
    perf = summarize_perf([], decode_only=False)
    assert perf.ttft_ms == 0.0
    assert perf.decode_total_time == 0.0
    assert perf.decode_total_tokens == 0
    assert perf.tokens_per_second == 0.0


# --------------------------------------------------------------------------- #
# build_compile_options
# --------------------------------------------------------------------------- #
def test_build_compile_options_required_only():
    opts = build_compile_options(
        CompileConfig(
            optimization_level=2,
            trace_enabled=True,
            experimental_weight_dtype="bfp_bf8",
            experimental_enable_permute_matmul_fusion=False,
            fp32_dest_acc_en=None,
            experimental_kv_cache_dtype=None,
            enable_create_d2m_subgraphs=False,
        ),
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
    )
    assert opts["optimization_level"] == 2
    assert opts["enable_trace"] is True
    assert opts["export_model_name"] == "m"
    # Optional keys must be absent when their inputs are None/False.
    assert "fp32_dest_acc_en" not in opts
    assert "experimental-kv-cache-dtype" not in opts
    assert "enable_create_d2m_subgraphs" not in opts


def test_build_compile_options_includes_optionals_when_set():
    opts = build_compile_options(
        CompileConfig(
            optimization_level=1,
            trace_enabled=False,
            experimental_weight_dtype="",
            experimental_enable_permute_matmul_fusion=True,
            fp32_dest_acc_en=True,
            experimental_kv_cache_dtype="bfp_bf8",
            enable_create_d2m_subgraphs=True,
        ),
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
    )
    assert opts["fp32_dest_acc_en"] is True
    assert opts["experimental-kv-cache-dtype"] == "bfp_bf8"
    assert opts["enable_create_d2m_subgraphs"] is True


# --------------------------------------------------------------------------- #
# evaluate_pcc
# --------------------------------------------------------------------------- #
def _logits(*rows):
    """Build a [len(rows), V] logits tensor where each row is a 1D sequence."""
    return torch.tensor([list(r) for r in rows], dtype=torch.float32)


def test_evaluate_pcc_passes_when_identical():
    # Two steps (prefill, decode), each a [1, V] tensor.
    out = [_logits([1.0, 2.0, 3.0, 4.0]), _logits([4.0, 3.0, 2.0, 1.0])]
    cpu = [t.clone() for t in out]
    # Identical -> PCC ~1.0 -> no assertion error.
    evaluate_pcc(
        out,
        cpu,
        required_pcc=0.99,
        decode_only=False,
        assert_prefill=True,
        assert_decode=True,
    )


def test_evaluate_pcc_raises_on_bad_prefill():
    out = [_logits([1.0, 2.0, 3.0, 4.0]), _logits([1.0, 2.0, 3.0, 4.0])]
    cpu = [_logits([4.0, 3.0, 2.0, 1.0]), _logits([1.0, 2.0, 3.0, 4.0])]
    with pytest.raises(AssertionError, match="Prefill PCC failed"):
        evaluate_pcc(
            out,
            cpu,
            required_pcc=0.99,
            decode_only=False,
            assert_prefill=True,
            assert_decode=False,
        )


def test_evaluate_pcc_decode_only_uses_cpu_index_1():
    # decode_only compares out[0] vs cpu[1]; cpu[0] is intentionally garbage.
    out = [_logits([1.0, 2.0, 3.0, 4.0])]
    cpu = [_logits([9.0, 9.0, 9.0, 9.0]), _logits([1.0, 2.0, 3.0, 4.0])]
    evaluate_pcc(
        out,
        cpu,
        required_pcc=0.99,
        decode_only=True,
        assert_prefill=False,
        assert_decode=True,
    )


def test_evaluate_pcc_skips_assert_when_flags_false():
    # Bad decode PCC but assert_decode=False -> must not raise.
    out = [_logits([1.0, 2.0, 3.0, 4.0]), _logits([1.0, 2.0, 3.0, 4.0])]
    cpu = [_logits([1.0, 2.0, 3.0, 4.0]), _logits([4.0, 3.0, 2.0, 1.0])]
    evaluate_pcc(
        out,
        cpu,
        required_pcc=0.99,
        decode_only=False,
        assert_prefill=True,
        assert_decode=False,
    )

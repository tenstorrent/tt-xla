# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vLLM accuracy helpers (no device / no vLLM engine)."""
from types import SimpleNamespace

import pytest
from benchmarks.vllm_benchmark import _extract_decode_predictions


def _logprob(lp):
    return SimpleNamespace(logprob=lp)


def _make_output(per_step_logprobs):
    # per_step_logprobs: list[dict[token_id -> logprob float]]
    steps = [
        {tid: _logprob(lp) for tid, lp in step.items()} for step in per_step_logprobs
    ]
    completion = SimpleNamespace(logprobs=steps)
    return SimpleNamespace(outputs=[completion])


def test_extracts_argmax_per_step():
    out = _make_output([{5: -0.1, 9: -2.0}, {7: -0.5}, {3: -0.2, 1: -0.3}])
    assert _extract_decode_predictions(out, 3) == [5, 7, 3]


def test_asserts_on_missing_logprobs():
    out = SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)])
    with pytest.raises(AssertionError):
        _extract_decode_predictions(out, 3)


def test_asserts_on_short_window():
    out = _make_output([{5: -0.1}, {7: -0.5}])
    with pytest.raises(AssertionError):
        _extract_decode_predictions(out, 3)

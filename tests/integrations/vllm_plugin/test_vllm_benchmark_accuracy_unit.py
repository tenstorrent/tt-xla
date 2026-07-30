# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vLLM accuracy-benchmark helpers (no device, no engine).

These cover code that lives in ``tests/benchmark`` but are collected from here:
``tests/benchmark`` is only ever invoked through explicit pytest node ids taken
from ``perf-bench-matrix.json``, so nothing under it runs as a directory and a
test file placed there would never execute in CI. This tree is collected by
mark, so ``push and cpu`` picks these up on the cpu runner.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# tests/integrations/vllm_plugin/<this file> -> tests/benchmark
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmarks.vllm_benchmark import _extract_decode_predictions  # noqa: E402


def _logprob(lp):
    return SimpleNamespace(logprob=lp)


def _make_output(per_step_logprobs):
    # per_step_logprobs: list[dict[token_id -> logprob float]]
    steps = [
        {tid: _logprob(lp) for tid, lp in step.items()} for step in per_step_logprobs
    ]
    completion = SimpleNamespace(logprobs=steps)
    return SimpleNamespace(outputs=[completion])


@pytest.mark.push
@pytest.mark.cpu
def test_extracts_argmax_per_step():
    out = _make_output([{5: -0.1, 9: -2.0}, {7: -0.5}, {3: -0.2, 1: -0.3}])
    assert _extract_decode_predictions(out, 3) == [5, 7, 3]


@pytest.mark.push
@pytest.mark.cpu
def test_asserts_on_missing_logprobs():
    out = SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)])
    with pytest.raises(AssertionError):
        _extract_decode_predictions(out, 3)


@pytest.mark.push
@pytest.mark.cpu
def test_asserts_on_short_window():
    out = _make_output([{5: -0.1}, {7: -0.5}])
    with pytest.raises(AssertionError):
        _extract_decode_predictions(out, 3)

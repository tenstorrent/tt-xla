# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared vocabulary for autoregressive text-generation benchmarks.

This is the thin layer between the domain-agnostic schema (``utils.py``) and the
independently-developed text-generation drivers (``llm_benchmark.py`` and
``vllm_benchmark.py``). It holds *only* what is identical for any text-gen
benchmark regardless of execution engine — currently the custom-measurement
constructors. Each driver still computes its own metrics (the torch-xla LLM
driver from on-device iteration timings, the vLLM driver from engine metrics)
and owns all of its engine-specific features; they merely agree here on how a
shared measurement is shaped.

Deliberately depends on nothing else so neither driver is coupled to the other.
"""

from typing import Any, Dict


def ttft_measurement(ttft_ms: float) -> Dict[str, Any]:
    """Custom-measurement entry for time-to-first-token (milliseconds)."""
    return {"measurement_name": "ttft", "value": ttft_ms, "target": -1}


def throughput_measurement(samples_per_sec: float) -> Dict[str, Any]:
    """Custom-measurement entry for decode throughput (tokens/samples per second)."""
    return {
        "measurement_name": "samples_per_sec",
        "value": samples_per_sec,
        "target": -1,
    }

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared conftest for vLLM generative tests."""
import psutil


def check_host_memory(model_name: str) -> float:
    """Assert child process RSS is below the known threshold for a model.

    Inspired by https://github.com/tenstorrent/tt-xla/issues/3611 where
    a vllm upgrade caused a ~3x host memory regression during compilation.

    Measures the current RSS of child processes (e.g. vLLM EngineCore)
    while they are still running. Call this after generation completes
    but before the engine is torn down.

    Returns the max child process RSS in GB.
    """
    # Known-good baselines with ~50% headroom.
    # Update these when adding new models or if baselines shift.
    model_rss_limits_gb = {
        "Qwen/Qwen3-0.6B": 5,
        "Qwen/Qwen3-1.7B": 9,
        "Qwen/Qwen3-4B": 19,
        "Qwen/Qwen3-32B": 150,
        "Qwen/Qwen2.5-32B": 105,
        "meta-llama/Llama-3.1-70B": 237,
    }

    # Measures max RSS across child processes; assumes one engine at a time.
    children = psutil.Process().children(recursive=True)
    rss_gb = max((c.memory_info().rss for c in children), default=0) / 1024**3
    threshold = model_rss_limits_gb.get(model_name)

    limit_str = f"{threshold} GB" if threshold is not None else "none set"
    print(f"[MEM] {model_name}: max child RSS = {rss_gb:.1f} GB (limit: {limit_str})")

    if threshold is not None:
        assert rss_gb < threshold, (
            f"Max child RSS {rss_gb:.1f} GB exceeds {threshold} GB "
            f"for {model_name} — possible host memory regression"
        )

    return rss_gb

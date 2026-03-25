# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Resilient vLLM fixtures for sampling tests.

Provides a cached LLM factory that auto-recreates the engine after a test
failure (which can leave the vLLM engine core in a dead state).
"""

import gc
import json
import os
import sys
from datetime import datetime, timezone

import pytest
import vllm

TEST_TIMEOUT_SECONDS = 180

_llm_cache: dict[str, vllm.LLM] = {}
_needs_recreate = False


def get_or_create_llm(name: str, **llm_args) -> vllm.LLM:
    """Return a cached LLM instance, recreating it if flagged unhealthy."""
    global _needs_recreate
    if name not in _llm_cache or _needs_recreate:
        _needs_recreate = False
        _flush_llm_cache()
        _llm_cache[name] = vllm.LLM(**llm_args)
    return _llm_cache[name]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Flag the LLM engine for recreation after any test failure."""
    global _needs_recreate
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        _needs_recreate = True


def _flush_llm_cache():
    """Delete all cached LLM instances and force GC to terminate their subprocesses."""
    for name in list(_llm_cache):
        llm = _llm_cache.pop(name)
        # Explicitly shut down the engine core subprocess before deleting the
        # LLM object.  Without this, the subprocess may still hold the TT
        # device when the next LLM is created, causing a hang.
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
    _llm_cache.clear()
    gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """Shut down cached vLLM engines when moving to a different test module.

    _llm_cache is intended to share engine instances within a single test
    module, not across modules.  Engines running in EngineCore subprocesses
    hold the TT device; if they outlive their module, any subsequent test that
    accesses the device directly (e.g. via torch.compile) will hang.
    """
    current_module = getattr(item, "module", None)
    next_module = getattr(nextitem, "module", None) if nextitem else None
    if current_module is not next_module and _llm_cache:
        _flush_llm_cache()


def pytest_addoption(parser):
    parser.addoption(
        "--iterations",
        type=int,
        default=50,
        help="Number of sampling iterations for perf benchmarks (default: 50)",
    )
    parser.addoption(
        "--perf-output",
        type=str,
        default=None,
        help="Path for perf benchmark JSON output (default: perf_debug/sampling_perf_<timestamp>.json)",
    )


@pytest.fixture
def iterations(request):
    return request.config.getoption("--iterations")


# -- Perf result collection --

_perf_results: list[dict] = []


@pytest.fixture
def perf_collector():
    """Append a result dict to the session-wide perf collector."""

    def record(
        test_name, backend, mode, vocab_size, iterations, avg_ms, p50_ms, p95_ms
    ):
        _perf_results.append(
            {
                "test": test_name,
                "backend": backend,
                "mode": mode,
                "vocab_size": vocab_size,
                "iterations": iterations,
                "avg_ms": round(avg_ms, 2),
                "p50_ms": round(p50_ms, 2),
                "p95_ms": round(p95_ms, 2),
                "tok_s": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0,
            }
        )

    return record


def pytest_sessionfinish(session, exitstatus):
    """Shut down cached LLM instances and write perf results."""
    _flush_llm_cache()

    if not _perf_results:
        return

    # Print summary table to stderr so it's visible even without -s
    w = sys.stderr.write
    header = (
        f"  {'Test':<35} {'Backend':<8} {'Vocab':>8} {'Avg ms':>8} "
        f"{'P50 ms':>8} {'P95 ms':>8} {'tok/s':>7}"
    )
    width = len(header)
    w("\n" + "=" * width + "\n")
    w("SAMPLING PERF SUMMARY\n")
    w("=" * width + "\n")
    w(header + "\n")
    w("-" * width + "\n")
    for r in _perf_results:
        w(
            f"  {r['test']:<35} {r['backend']:<8} {r['vocab_size']:>8} "
            f"{r['avg_ms']:>8.2f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} "
            f"{r['tok_s']:>7.1f}\n"
        )
    w("=" * width + "\n")

    # Write JSON
    output_path = session.config.getoption("--perf-output", default=None)
    if output_path is None:
        os.makedirs("perf_debug", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"perf_debug/sampling_perf_{ts}.json"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": session.config.getoption("--iterations", default=100),
        "results": _perf_results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write(f"\nPerf results written to: {output_path}\n")


@pytest.fixture
def prompt():
    return ["Once upon a time, there was a"]

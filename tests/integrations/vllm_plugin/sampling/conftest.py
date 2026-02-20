# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Resilient vLLM fixtures for sampling tests.

Provides a cached LLM factory that auto-recreates the engine after a test
failure (which can leave the vLLM engine core in a dead state).
"""

import gc

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


def _flush_llm_cache():
    """Delete all cached LLM instances and force GC to terminate their subprocesses."""
    for name in list(_llm_cache):
        del _llm_cache[name]
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


def pytest_sessionfinish(session, exitstatus):
    """Shut down cached LLM instances before Python tears down I/O."""
    _flush_llm_cache()


@pytest.fixture
def prompt():
    return ["Once upon a time, there was a"]

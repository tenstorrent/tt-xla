# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Resilient vLLM fixtures for sampling tests.

Provides a cached LLM factory that auto-recreates the engine after a test
failure (which can leave the vLLM engine core in a dead state).
"""

import pytest
import vllm

TEST_TIMEOUT_SECONDS = 120

_llm_cache: dict[str, vllm.LLM] = {}
_needs_recreate = False


def get_or_create_llm(name: str, **llm_args) -> vllm.LLM:
    """Return a cached LLM instance, recreating it if flagged unhealthy."""
    global _needs_recreate
    if name not in _llm_cache or _needs_recreate:
        _needs_recreate = False
        _llm_cache[name] = vllm.LLM(**llm_args)
    return _llm_cache[name]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Flag all cached LLMs for recreation if a test fails or errors."""
    global _needs_recreate
    outcome = yield
    report = outcome.get_result()
    if report.when in ("call", "setup") and report.failed:
        _needs_recreate = True


@pytest.fixture
def prompt():
    return ["Once upon a time, there was a"]

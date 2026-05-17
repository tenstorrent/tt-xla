# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared cleanup for vLLM-plugin tests.

vLLM has no public LLM.shutdown(); engine teardown runs via
weakref.finalize when the LLM becomes unreachable. On a failed test,
pytest's report pins the test-frame locals (incl. `llm`) via the live
traceback, so finalize never fires and the EngineCore subprocess
holding the TT device outlives the test — hanging the next one. Shut
them down explicitly on failure.
"""
import gc

import pytest
import vllm


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    for obj in gc.get_objects():
        try:
            is_llm = isinstance(obj, vllm.LLM)
        except ReferenceError:
            # Dead weakref proxies in gc.get_objects() raise here.
            continue
        if not is_llm:
            continue
        try:
            obj.llm_engine.engine_core.shutdown()
        except Exception:
            pass

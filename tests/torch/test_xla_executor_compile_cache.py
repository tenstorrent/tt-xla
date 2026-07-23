# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for XLAExecutor's experimental compiled-graph cache.

The non-AOT (experimental) compile path caches the torch-xla compiled graph per
XLAExecutor. A single executor can be invoked with more than one input shape
(e.g. an LLM's prefill seq_len vs decode seq_len=1 reusing the same dynamo
graph). The graph returned by ``bridge.extract_compiled_graph`` is specialized
to the shapes it was extracted with, so it must be cached *per shape*; reusing a
prefill-shaped graph for a decode call returns a stale, wrongly-shaped result.

This test mocks ``extract_compiled_graph`` (no device needed) and asserts the
cache extracts one graph per distinct input shape and reuses it for repeats.
"""

import pytest
import torch
import tt_torch.backend.backend as backend_mod
from tt_torch.backend.backend import XLAExecutor


def _make_executor():
    # Bypass the heavy __init__ (needs a real GraphModule/signature); we only
    # exercise the experimental compile-cache logic.
    ex = XLAExecutor.__new__(XLAExecutor)
    ex.compiled_graphs = {}
    ex.module = object()
    ex.params_and_consts = ()
    return ex


def test_experimental_compile_cache_keys_by_shape(monkeypatch):
    extractions = []

    def fake_extract(module, full_args):
        # A real extracted graph is specialized to these shapes; mimic that by
        # returning a callable that echoes the shapes it was built for (and
        # ignores the shapes it is later called with).
        shapes = tuple(tuple(a.shape) for a in full_args if isinstance(a, torch.Tensor))
        extractions.append(shapes)
        return lambda *args: shapes

    monkeypatch.setattr(backend_mod.bridge, "extract_compiled_graph", fake_extract)

    ex = _make_executor()
    prefill = (torch.zeros(32, 128, dtype=torch.long),)
    decode = (torch.zeros(32, 1, dtype=torch.long),)

    # Prefill: one extraction, graph specialized to (32, 128).
    assert ex._call_experimental_compile(prefill) == ((32, 128),)
    # Repeat prefill: cache hit, no new extraction.
    assert ex._call_experimental_compile(prefill) == ((32, 128),)
    # Decode (new shape): must extract a fresh graph specialized to (32, 1),
    # NOT reuse the prefill graph (which would wrongly return (32, 128)).
    assert ex._call_experimental_compile(decode) == ((32, 1),)
    # Repeat decode: cache hit.
    assert ex._call_experimental_compile(decode) == ((32, 1),)

    assert extractions == [((32, 128),), ((32, 1),)], "one extraction per shape"
    assert len(ex.compiled_graphs) == 2


def test_experimental_compile_cache_distinguishes_dtype(monkeypatch):
    extractions = []

    def fake_extract(module, full_args):
        extractions.append(tuple(a.dtype for a in full_args))
        return lambda *args: "graph"

    monkeypatch.setattr(backend_mod.bridge, "extract_compiled_graph", fake_extract)

    ex = _make_executor()
    ex._call_experimental_compile((torch.zeros(4, dtype=torch.float32),))
    ex._call_experimental_compile((torch.zeros(4, dtype=torch.bfloat16),))
    # Same shape, different dtype => two distinct graphs.
    assert len(ex.compiled_graphs) == 2
    assert len(extractions) == 2

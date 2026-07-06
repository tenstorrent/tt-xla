# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the mixed_precision CPU tests.

Two things are set up here:

1. Custom CLI options the tt-xla CI harness injects. mixed_precision/ lives outside
   tests/, so tests/conftest.py — which normally registers these — is not loaded when
   CI runs `pytest ./mixed_precision`. Without registration pytest aborts with
   "unrecognized arguments: --log-memory" before collecting anything.

2. A mock TT cluster descriptor. The sensitivity-score path calls
   ttnn.typecast(..., bfloat4_b), whose host-side BFP4 packing is pure CPU math but
   reads one L1-alignment constant through MetalContext. MetalContext::instance()
   eagerly builds the silicon cluster, which aborts with "No chips detected" on a
   device-less runner (the CI cpu runner). Pointing TT_METAL_MOCK_CLUSTER_DESC_PATH at
   a cluster-descriptor YAML shipped inside the pjrt_plugin_tt wheel makes the runtime
   load the descriptor from file instead of probing hardware, so host typecast runs
   with no chip present. Must be set before the first ttnn op / MetalContext init.
"""

import glob
import importlib.util
import os


def _enable_mock_cluster():
    """Point TT_METAL_MOCK_CLUSTER_DESC_PATH at a descriptor shipped in the wheel.

    No-op if already set (respects an externally provided descriptor) or if the wheel /
    descriptor can't be found (the ttnn test then skips via its own importorskip guard).
    """
    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        return
    spec = importlib.util.find_spec("pjrt_plugin_tt")
    if spec is None or spec.origin is None:
        return
    examples = os.path.join(
        os.path.dirname(spec.origin),
        "tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples",
    )
    preferred = os.path.join(examples, "wormhole_N150.yaml")
    candidates = (
        [preferred]
        if os.path.exists(preferred)
        else sorted(glob.glob(os.path.join(examples, "*.yaml")))
    )
    if candidates:
        os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = candidates[0]


_enable_mock_cluster()


def pytest_addoption(parser):
    parser.addoption("--log-memory", action="store_true", default=False)
    parser.addoption("--perf-report-dir", action="store", default=None)
    parser.addoption("--perf-id", action="store", default=None)
    parser.addoption("--dump-irs", action="store_true", default=False)
    parser.addoption("--arch", action="store", default=None)

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA vision sanity tests compile a different graph per stack depth (26+ variants).

PyTorch Dynamo's default recompile_limit (8) causes silent graph degradation after a few
depths — PCC can flip negative / near zero, which is *not* real numerical drift (~0.94).

Raise the limit for this directory so layer-wise PCC reflects device numerics, not dynamo.
"""

from __future__ import annotations

import pytest

_MIN_RECOMPILE_LIMIT = 64


@pytest.fixture(scope="session", autouse=True)
def _llava_sanity_dynamo_recompile_limit():
    try:
        import torch._dynamo.config as dynamo_config
    except ImportError:
        yield
        return

    prev = dynamo_config.recompile_limit
    dynamo_config.recompile_limit = max(prev, _MIN_RECOMPILE_LIMIT)
    yield
    dynamo_config.recompile_limit = prev

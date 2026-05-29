# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B — UMT5-XXL Text Encoder component test.

A14B and 5B share the same UMT5-XXL encoder config and weights — this test
is structurally identical to wan5b's text-encoder test, but loads from the
A14B repo (see ``shared.MODEL_ID``).

IN:  input_ids (1, 512) int64, attention_mask (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) float
"""

from typing import Optional

import pytest
import torch
from infra import ComparisonConfig, Framework, run_graph_test
from infra.evaluators import PccConfig
from infra.utilities import Mesh

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import (
    UMT5Wrapper,
    load_umt5,
    shard_umt5_specs,
    wan22_mesh,
)

_COMPILER_CONFIG = CompilerConfig(
    optimization_level=1,
    enable_trace=True,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_umt5_480p_sharded():
    _run(sharded=True)


def _run(sharded: bool) -> None:
    torch.manual_seed(42)
    wrapper = UMT5Wrapper(load_umt5()).eval().bfloat16()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_umt5_specs(m.encoder)) if sharded else None

    run_graph_test(
        graph=wrapper,
        inputs=[input_ids, attention_mask],
        framework=Framework.TORCH,
        compiler_config=_COMPILER_CONFIG,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
    )

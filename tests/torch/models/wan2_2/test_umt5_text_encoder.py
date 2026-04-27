# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — UMT5-XXL Text Encoder component test.

IN:  input_ids (1, 512) int64, attention_mask (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) float
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import RESOLUTIONS, load_umt5, shard_umt5_specs, wan22_mesh


class UMT5Wrapper(torch.nn.Module):
    """Return last_hidden_state as a plain tensor (not a model output object)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def test_umt5_480p():  # OOM on single device
    _run(resolution="480p", sharded=False)


def test_umt5_720p():  # OOM on single device
    _run(resolution="720p", sharded=False)


def test_umt5_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_umt5_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)
    _ = RESOLUTIONS[resolution]  # resolution is a no-op for UMT5 shapes

    wrapper = UMT5Wrapper(load_umt5()).eval().bfloat16()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_umt5_specs(m.encoder)) if sharded else None

    run_graph_test(
        wrapper,
        [input_ids, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )

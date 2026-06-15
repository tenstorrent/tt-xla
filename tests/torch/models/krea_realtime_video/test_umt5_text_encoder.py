# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
krea-realtime-video — UMT5-XXL Text Encoder component test.

Same UMT5EncoderModel as Wan 2.1/2.2 (loaded from the Wan 2.1 diffusers mirror).
~5.5B params (~11 GB bf16): OOMs on a single n150 (12 GiB, ~85% weight budget),
so the single-device nodes are expected to be weight-bound — bring it up
tensor-parallel on the mesh.

IN:  input_ids (1, 512) int64, attention_mask (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) float
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import krea_mesh, load_umt5, shard_umt5_specs


class UMT5Wrapper(torch.nn.Module):
    """Return last_hidden_state as a plain tensor (not a model output object)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


@pytest.mark.nightly
@pytest.mark.model_test
def test_umt5_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    wrapper = UMT5Wrapper(load_umt5()).eval().bfloat16()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)

    mesh = krea_mesh() if sharded else None
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

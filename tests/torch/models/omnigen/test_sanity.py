# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — AutoencoderKLHunyuanVideo decoder component test."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator


class OmniGenSuScaledRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        base=10000,
        rope_scaling=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

        self.short_factor = rope_scaling["short_factor"]
        self.long_factor = rope_scaling["long_factor"]
        self.original_max_position_embeddings = original_max_position_embeddings

    def forward(self, inv_freq_expanded, position_ids_expanded):
        with torch.autocast(device_type="cpu", enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)[0]

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1
                    + math.log(scale) / math.log(self.original_max_position_embeddings)
                )

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos, sin


@pytest.mark.nightly
@pytest.mark.model_test
def test_rope_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    model = (
        OmniGenSuScaledRotaryEmbedding(
            dim=96,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            base=10000,
            rope_scaling={
                "long_factor": [
                    1.0299999713897705,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0799999237060547,
                    1.2299998998641968,
                    1.2299998998641968,
                    1.2999999523162842,
                    1.4499999284744263,
                    1.5999999046325684,
                    1.6499998569488525,
                    1.8999998569488525,
                    2.859999895095825,
                    3.68999981880188,
                    5.419999599456787,
                    5.489999771118164,
                    5.489999771118164,
                    9.09000015258789,
                    11.579999923706055,
                    15.65999984741211,
                    15.769999504089355,
                    15.789999961853027,
                    18.360000610351562,
                    21.989999771118164,
                    23.079999923706055,
                    30.009998321533203,
                    32.35000228881836,
                    32.590003967285156,
                    35.56000518798828,
                    39.95000457763672,
                    53.840003967285156,
                    56.20000457763672,
                    57.95000457763672,
                    59.29000473022461,
                    59.77000427246094,
                    59.920005798339844,
                    61.190006256103516,
                    61.96000671386719,
                    62.50000762939453,
                    63.3700065612793,
                    63.48000717163086,
                    63.48000717163086,
                    63.66000747680664,
                    63.850006103515625,
                    64.08000946044922,
                    64.760009765625,
                    64.80001068115234,
                    64.81001281738281,
                    64.81001281738281,
                ],
                "short_factor": [
                    1.05,
                    1.05,
                    1.05,
                    1.1,
                    1.1,
                    1.1,
                    1.2500000000000002,
                    1.2500000000000002,
                    1.4000000000000004,
                    1.4500000000000004,
                    1.5500000000000005,
                    1.8500000000000008,
                    1.9000000000000008,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.1000000000000005,
                    2.1000000000000005,
                    2.2,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3999999999999995,
                    2.3999999999999995,
                    2.6499999999999986,
                    2.6999999999999984,
                    2.8999999999999977,
                    2.9499999999999975,
                    3.049999999999997,
                    3.049999999999997,
                    3.049999999999997,
                ],
                "type": "su",
            },
        )
        .to(torch.bfloat16)
        .eval()
    )
    inputs = [
        torch.load("./inv_freq_expanded.pt"),
        torch.load("./position_ids_expanded.pt"),
    ]

    with _mask_jax_accelerator():
        run_graph_test(model, inputs, framework=Framework.TORCH)

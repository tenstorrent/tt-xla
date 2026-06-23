# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Self-contained sanity reproducing the device output explosion found during
hexgrad/Kokoro-82M bringup.

Failing op
----------
``nn.InstanceNorm1d(num_features, affine=True)`` inside the iSTFTNet vocoder's
AdaIN1d normalization. Device-vs-CPU bisect localized the Kokoro output
explosion (atol ~2e38, PCC nan) to the FIRST AdaIN1d inside
Generator.noise_res[0] (an AdaINResBlock1).

Root cause
----------
InstanceNorm normalizes each channel as ``(x - mean) / sqrt(var + eps)``. When
its input is a NEAR-CONSTANT activation produced *in-graph* by a preceding conv
(O(1) per-channel offset, tiny per-channel variance ~1e-7), the bf16 variance
(``E[x^2] - E[x]^2`` / the ``x - mean`` subtraction) catastrophically cancels to
~0 / slightly negative -> ``rsqrt`` -> ``inf`` (clamped to FLT_MAX 3.4e38) /
``nan``. The fp32 CPU reference keeps the small-but-finite variance and stays
finite, so device-vs-CPU PCC collapses to ``nan``.

Why the conv must be in-graph
-----------------------------
A bare ``InstanceNorm1d`` fed a near-constant tensor *as a graph input* does NOT
reproduce (see test_adain_sanity.py -> PASSES): the trigger needs the conv
producing the near-constant tensor as a bf16 intermediate. This sanity therefore
chains ``Conv1d -> AdaIN1d`` and constructs the conv to emit a near-constant
output deterministically (tiny weights + O(1) per-channel bias), matching the
random-weight-Kokoro regime (the untrained noise-branch conv emits a
near-constant activation; trained weights carry real variance and the path is
well conditioned).
"""

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

# Matches Kokoro Generator.noise_convs[0]: Conv1d(22, 256, k=12, s=6, p=3) ->
# AdaIN1d over 256 channels with a 128-wide style vector.
_IN_CH = 22
_NUM_FEATURES = 256
_KERNEL = 12
_STRIDE = 6
_PAD = 3
_STYLE_DIM = 128
_LENGTH = 30001  # ~ Kokoro har length at the bringup token count


@pytest.mark.single_device
@pytest.mark.xfail(
    reason="bf16 InstanceNorm variance catastrophic cancellation on a near-constant "
    "conv output -> rsqrt -> inf/nan; device numerical-robustness gap (see <ISSUE_URL>).",
    strict=False,
)
def test_conv_adain1d_near_constant_explodes():

    class ConvAdaIN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(
                _IN_CH, _NUM_FEATURES, _KERNEL, stride=_STRIDE, padding=_PAD
            )
            self.norm = torch.nn.InstanceNorm1d(_NUM_FEATURES, affine=True)
            self.fc = torch.nn.Linear(_STYLE_DIM, _NUM_FEATURES * 2)
            # Make the conv emit a NEAR-CONSTANT activation: tiny weights so the
            # output is dominated by an O(1) per-channel bias, leaving a per-
            # channel variance ~1e-7 (the pathological regime). Deterministic.
            torch.manual_seed(0)
            with torch.no_grad():
                self.conv.weight.copy_(torch.randn_like(self.conv.weight) * 1e-5)
                self.conv.bias.copy_(torch.linspace(0.5, 1.5, _NUM_FEATURES))

        def forward(self, har, s):
            x = self.conv(har)
            h = self.fc(s).view(s.size(0), -1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta

    model = ConvAdaIN()
    model.eval()

    torch.manual_seed(0)
    # Bounded input resembling the vocoder's har (STFT magnitude + phase).
    har = (torch.rand(1, _IN_CH, _LENGTH) * 2 - 1) * torch.pi
    s = torch.randn(1, _STYLE_DIM)

    with torch.no_grad():
        conv_out = model.conv(har)
    logger.info(
        "conv_out: abs.max={:.4f}, min per-channel var={:.3e}",
        conv_out.abs().max().item(),
        conv_out.var(dim=2, unbiased=False).min().item(),
    )

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(framework=Framework.TORCH, model=model, args=[har, s])
    tester.test(workload)

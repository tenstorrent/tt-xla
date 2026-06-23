# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal self-contained reproducer of the InstanceNorm bf16-variance explosion
found during hexgrad/Kokoro-82M bringup.

Failing op
----------
``nn.InstanceNorm1d(num_features, affine=True)`` (the core of the iSTFTNet
vocoder's AdaIN1d). Device-vs-CPU bisect localized the Kokoro output explosion
(atol ~2e38, PCC nan) to the first AdaIN1d inside Generator.noise_res[0].

Root cause
----------
InstanceNorm normalizes each channel as ``(x - mean) / sqrt(var + eps)``. On a
NEAR-CONSTANT channel (O(1) per-channel mean, tiny per-channel variance) the
bf16 statistics (``E[x^2] - E[x]^2`` / the ``x - mean`` subtraction) suffer
catastrophic cancellation: with ~3 significant bf16 figures, ``E[x^2]`` and
``E[x]^2`` round to the same value, so the computed variance collapses to ~0 /
slightly negative -> ``rsqrt`` -> ``inf`` (clamped to FLT_MAX 3.4e38) / ``nan``.
The fp32 CPU reference keeps the small-but-finite variance and stays finite, so
device-vs-CPU PCC collapses to ``nan``. Severity scales with the mean/std
ratio: a healthy-variance input normalizes cleanly.

In Kokoro this is triggered by random/untrained weights (the noise-branch conv
emits a near-constant activation); trained weights carry real variance and the
path is well conditioned. See test_adain_chain_sanity.py for the same failure
through the actual Conv1d -> AdaIN1d structure.

xfail: tracks an open device numerical-robustness gap (see <ISSUE_URL>).
"""

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

_NUM_FEATURES = 256
_STYLE_DIM = 128
_LENGTH = 5000


@pytest.mark.single_device
@pytest.mark.xfail(
    reason="bf16 InstanceNorm variance catastrophic cancellation on near-constant "
    "input -> rsqrt -> inf/nan; device numerical-robustness gap (see <ISSUE_URL>).",
    strict=False,
)
def test_instancenorm1d_near_constant_bf16_explodes():

    class AdaIN1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.InstanceNorm1d(_NUM_FEATURES, affine=True)
            self.fc = torch.nn.Linear(_STYLE_DIM, _NUM_FEATURES * 2)

        def forward(self, x, s):
            h = self.fc(s).view(s.size(0), -1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta

    model = AdaIN1d()
    model.eval()

    torch.manual_seed(0)
    # Near-constant per-channel input: O(1) offset + ~1e-4 jitter (var ~2.6e-8,
    # mean/std ratio ~6000:1) -> bf16 variance cancels.
    base = torch.linspace(0.5, 1.5, _NUM_FEATURES).view(1, _NUM_FEATURES, 1)
    x = base + torch.randn(1, _NUM_FEATURES, _LENGTH) * 1.6e-4
    s = torch.randn(1, _STYLE_DIM)

    logger.info(
        "x: abs.max={:.4f}, min per-channel var={:.3e}",
        x.abs().max().item(),
        x.var(dim=2, unbiased=False).min().item(),
    )

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(framework=Framework.TORCH, model=model, args=[x, s])
    tester.test(workload)

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Repro for a suspected ``ttnn.softmax`` kernel bug on causal-masked attention scores.

Context
-------
DeepSeek-V3.1 (tt-xla benchmark) shows every attention ``ttnn.softmax`` failing chisel
numerics in *isolated* mode (PCC ~= 0, atol ~= 1.0) across all layers, prefill and decode.
In isolated mode chisel feeds the *same* device input tensor into both the device op and the
torch golden, so the divergence is in the softmax op itself, given identical input.

The failing softmax input is a causal-masked score tensor whose masked positions are set to
``torch.finfo(torch.bfloat16).min`` (== -3.38953139e38, ~= -FLT_MAX); see the model graph:

    %68  = ttnn.full{fill_value = -3.38953139e38}        # bf16 finfo.min sentinel
    %197 = ttnn.multiply(causal_mask, %68)               # mask * -3.39e38
    %198 = ttnn.add(scaled_scores, %197)                 # scores + mask  (bf16)
    %199 = ttnn.typecast(%198 -> f32)
    %200 = ttnn.softmax(%199, dim=3, numericStable=true)  # <-- diverges from golden

Hypothesis: the device softmax mishandles the near -FLT_MAX masked entries (overflow in the
``x - rowmax`` / ``exp`` / normalization path), collapsing the output distribution.

What this test does
-------------------
Builds ``softmax(masked_scores)`` where the input is provided verbatim (via ``set_goldens``) as a
causal-masked f32 score tensor, and compares the device output against ``torch.softmax`` (PCC).
It mirrors the model's compute config (math_fidelity=hifi4, fp32_dest_acc_en=true) and op
(dim=-1, numericStable).

Expected outcome (the point of the repro)
-----------------------------------------
* ``mask_value = bf16_min`` (-3.38953139e38): EXPECTED TO FAIL with low PCC -> reproduces the bug.
* ``mask_value = -1e4``:                       EXPECTED TO PASS -> a milder sentinel still drives
  masked probabilities to ~0 but avoids the overflow class. If true, the frontend fix is to clamp
  the mask sentinel instead of emitting ``finfo(bf16).min``.

How to run (inside a built tt-mlir checkout)
--------------------------------------------
    source env/activate
    ttrt query --save-artifacts
    export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
    pytest -svv test/python/golden/test_softmax_masked_min_repro.py --sys-desc=$SYSTEM_DESC_PATH
"""
from typing import List, Optional

import pytest
import torch
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

# bf16 finfo.min, exactly as emitted by the model's mask constant (%68).
BF16_MIN = float(torch.finfo(torch.bfloat16).min)  # -3.3895313892515355e+38


def _causal_masked_scores(shape, mask_value: float, dtype: torch.dtype) -> torch.Tensor:
    """A [..., S, S] tensor of small random scores with the strict upper triangle masked.

    The diagonal stays valid, so every softmax row has >= 1 finite entry (golden has no NaN).
    The masked region spans whole tiles, exercising the cross-tile reduction.
    """
    torch.manual_seed(0)
    scores = (torch.randn(shape) * 4.0).to(
        dtype
    )  # ~N(0, 4): realistic score magnitudes
    s = shape[-1]
    # upper-triangular (key_pos > query_pos) == future positions == masked
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, mask_value)
    return scores.to(dtype)


@pytest.mark.parametrize(
    "mask_value",
    [
        pytest.param(
            BF16_MIN, id="mask_bf16_min"
        ),  # reproduces the bug (expect low PCC)
        pytest.param(-1e4, id="mask_neg_1e4"),  # control: milder sentinel (expect pass)
    ],
)
@pytest.mark.parametrize("numeric_stable", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float32, id="f32"
        ),  # faithful: model casts to f32 before softmax
        pytest.param(torch.bfloat16, id="bf16"),
    ],
)
# (1, 8, 128, 128): one batch, 8 heads, 128x128 causal scores; dim 128 == 4 tiles of 32.
@pytest.mark.parametrize("shape", [(1, 8, 128, 128)])
def test_softmax_masked_min(request, device, shape, dtype, numeric_stable, mask_value):
    masked_scores = _causal_masked_scores(shape, mask_value, dtype)
    # Golden computed in fp32 for stability, cast back to the op's dtype.
    golden_output = torch.softmax(masked_scores.float(), dim=-1).to(dtype)

    def test_module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def softmax_masked(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            out = builder.softmax(
                in0,
                dimension=-1,
                numeric_stable=numeric_stable,
                unit_attrs=unit_attrs,
            )
            # Inject the exact masked input (also fed to the device) and the expected output.
            builder.set_goldens({in0: masked_scores}, {out: golden_output})
            return out

    # Match the model's attention softmax compute config.
    pipeline_options = [
        "compute-cfg-math-fidelity=hifi4",
        "compute-cfg-fp32-dest-acc-en=true",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )

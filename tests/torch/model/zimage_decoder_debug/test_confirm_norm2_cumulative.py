# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Three-test confirmation that Z-Image decoder OOM is cumulative at resnet0.norm2.

1. ``test_confirm_norm2_alone_passes`` — norm2 only, golden input after conv1.
   Expect: PASS (op is fine in isolation).

2. ``test_confirm_prefix_through_conv1_passes`` — prefix + resnet0 through conv1.
   Expect: PASS (cumulative graph still OK immediately before norm2).

3. ``test_confirm_prefix_through_norm2_ooms`` — prefix + resnet0 through norm2.
   Expect: RuntimeError OOM (~3.77 GiB subtract) — same failure class as full decoder.

Run:
  pytest -svv tests/torch/model/zimage_decoder_debug/test_confirm_norm2_cumulative.py 2>&1 | tee zimage_decoder_logs/confirm_norm2_cumulative.log
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from .shared import (
    build_confirm_prefix_through_conv1,
    build_confirm_prefix_through_norm2,
    build_norm2_only,
    d3_input_key,
    norm2_isolated_input_key,
)


@pytest.mark.model_test
def test_confirm_norm2_alone_passes(vae_decoder_context):
    """norm2 alone on TT must pass — op/lowering is not fundamentally broken."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_norm2_only(dec).eval()
    inputs = [stages[norm2_isolated_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_confirm_prefix_through_conv1_passes(vae_decoder_context):
    """Cumulative graph through conv1 (before norm2) must still pass."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_confirm_prefix_through_conv1(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_confirm_prefix_through_norm2_ooms(vae_decoder_context):
    """Cumulative graph through norm2 must OOM — same failure as prefix_through_up_4 / full decoder."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_confirm_prefix_through_norm2(dec).eval()
    inputs = [stages[d3_input_key()]]

    with pytest.raises(RuntimeError):
        run_graph_test(module, inputs, framework=Framework.TORCH)

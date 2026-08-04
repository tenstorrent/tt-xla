# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the fused ttml training composites.

These cover the frontend half of the tt-mlir additions in
  - tenstorrent/tt-mlir#9152 (ttir.sdpa_fw / ttnn.sdpa_fw)
  - tenstorrent/tt-mlir#9145 (ttir.cross_entropy_fw / ttnn.cross_entropy_fw)

Each test builds the composite directly (no auto-patching, since neither op has
a matching torch API contract), runs it on device against the CPU golden, and
FileChecks the serialized TTNN IR so a silent decomposition back into generic
ops is a failure rather than a pass.

Both ops need a tt-mlir containing the two PRs above; against an older pin they
fail with the composite left unlegalized.
"""

import pytest
import torch
from infra.utilities.types import Framework
from tt_torch.composite_ops import (
    AttentionMaskType,
    composite_cross_entropy_fw,
    composite_sdpa_fw,
)
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


def _additive_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """[1, 1, S, S] additive mask, the only shape ttir.sdpa_fw accepts."""
    return torch.triu(
        torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=dtype),
        diagonal=1,
    )


class _SDPAForward(torch.nn.Module):
    def __init__(self, mask_type: AttentionMaskType, return_intermediates: bool):
        super().__init__()
        self.mask_type = mask_type
        self.return_intermediates = return_intermediates

    def forward(self, query, key, value, attention_mask=None):
        return composite_sdpa_fw(
            query,
            key,
            value,
            attention_mask=attention_mask,
            mask_type=self.mask_type,
            return_intermediates=self.return_intermediates,
        )


class _CrossEntropyForward(torch.nn.Module):
    def forward(self, logits, target):
        return composite_cross_entropy_fw(logits, target)


def _sdpa_fw_inputs(batch, q_heads, kv_heads, seq_len, head_dim, mask_type):
    """Q/K/V (and mask) for the given layout; bf16 is the op's native dtype."""
    query = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

    inputs = [query, key, value]
    if mask_type == AttentionMaskType.ARBITRARY:
        inputs.append(_additive_causal_mask(seq_len, torch.bfloat16))
    return inputs


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.scaled_dot_product_attention",
    shlo_op_name="stablehlo.composite",
)
@pytest.mark.filecheck(["sdpa_fw.ttnn.mlir"])
# Causal first: (1, 8, 64, 64) bf16 causal is the only configuration PR 9152
# executes on device (test_ttir_ops.py::test_sdpa_fw), so it is the case most
# likely to work. none/arbitrary are exercised only by that PR's IR-level tests.
@pytest.mark.parametrize(
    "mask_type",
    [AttentionMaskType.CAUSAL, AttentionMaskType.NONE, AttentionMaskType.ARBITRARY],
    ids=["mask_causal", "mask_none", "mask_arbitrary"],
)
@pytest.mark.parametrize(
    "batch, q_heads, kv_heads, seq_len, head_dim",
    [
        (1, 8, 8, 64, 64),
        # GQA: Hq must be a positive multiple of Hkv.
        (1, 8, 2, 64, 64),
    ],
    ids=["1x8x64x64", "gqa_8q_2kv"],
)
def test_composite_sdpa_fw(
    request, mask_type, batch, q_heads, kv_heads, seq_len, head_dim
):
    options = {"tt_enable_composite_ops": False}

    model = _SDPAForward(mask_type, return_intermediates=False)
    inputs = _sdpa_fw_inputs(batch, q_heads, kv_heads, seq_len, head_dim, mask_type)

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            inputs,
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
            request=request,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.scaled_dot_product_attention",
    shlo_op_name="stablehlo.composite",
)
@pytest.mark.filecheck(["sdpa_fw_intermediates.ttnn.mlir"])
@pytest.mark.parametrize(
    "mask_type",
    [AttentionMaskType.CAUSAL, AttentionMaskType.ARBITRARY],
    ids=["mask_causal", "mask_arbitrary"],
)
def test_composite_sdpa_fw_intermediates(request, mask_type):
    """Two-result form: attention output plus the [B, Hq, S, 32] fp32 log-sum-exp."""
    batch, q_heads, seq_len, head_dim = 1, 8, 64, 64
    options = {"tt_enable_composite_ops": False}

    model = _SDPAForward(mask_type, return_intermediates=True)
    inputs = _sdpa_fw_inputs(batch, q_heads, q_heads, seq_len, head_dim, mask_type)

    def comparator(device_output, golden_output, args, kwargs):
        device_out, device_intermediates = (t.cpu() for t in device_output)
        golden_out, golden_intermediates = golden_output

        assert device_intermediates.shape == (batch, q_heads, seq_len, 32)
        assert device_intermediates.dtype == torch.float32

        for name, device, golden in (
            ("output", device_out, golden_out),
            ("intermediates", device_intermediates, golden_intermediates),
        ):
            pcc = torch.corrcoef(
                torch.stack([device.flatten().float(), golden.flatten().float()])
            )[0, 1]
            assert pcc > 0.99, f"{name} PCC: {pcc.item()} (required > 0.99)"

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            inputs,
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
            request=request,
            custom_comparator=comparator,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.cross_entropy",
    shlo_op_name="stablehlo.composite",
)
@pytest.mark.filecheck(["cross_entropy_fw.ttnn.mlir"])
@pytest.mark.parametrize(
    "batch, num_rows, num_classes",
    [
        (4, 32, 64),
        # A class count that is not tile-aligned takes the kernel's masked path.
        (2, 32, 100),
    ],
    ids=["4x1x32x64", "2x1x32x100"],
)
def test_composite_cross_entropy_fw(request, batch, num_rows, num_classes):
    options = {"tt_enable_composite_ops": False}

    logits = torch.randn(batch, 1, num_rows, num_classes, dtype=torch.bfloat16)
    # Indices must be valid classes, i.e. in [0, num_classes).
    target = torch.randint(0, num_classes, (batch, num_rows), dtype=torch.int32)

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            _CrossEntropyForward(),
            [logits, target],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
            request=request,
        )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.scaled_dot_product_attention",
    shlo_op_name="stablehlo.composite",
)
def test_composite_sdpa_fw_rejects_mask_type_mismatch():
    """The mask operand and mask_type must agree; tt-mlir rejects the composite otherwise."""
    query = key = value = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16)
    mask = _additive_causal_mask(64, torch.bfloat16)

    with pytest.raises(ValueError, match="attention_mask must be provided iff"):
        composite_sdpa_fw(
            query, key, value, attention_mask=mask, mask_type=AttentionMaskType.CAUSAL
        )

    with pytest.raises(ValueError, match="attention_mask must be provided iff"):
        composite_sdpa_fw(query, key, value, mask_type=AttentionMaskType.ARBITRARY)

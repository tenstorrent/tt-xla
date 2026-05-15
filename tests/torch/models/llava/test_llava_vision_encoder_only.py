# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP vision encoder only (first N CLIPEncoder layers), loaded from LLaVA.

Input: activations immediately before CLIPEncoder (post embeddings + pre_layernorm),
from fixture or randn with same shape.

Runs depths 1..24 with PCC threshold 0.99 for every depth; executes all depths then
reports failures (same pattern as test_llava_vision_through_encoder.py).
Dynamo recompile_limit is raised in conftest.py — see note there if PCC goes negative mid-stack.

Requires: tests/torch/models/llava/fixtures/encoder_pre_input_bf16.pt
"""

from __future__ import annotations

import pytest
from infra import Framework, run_op_test
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig
from tests.torch.models.llava.encoder_utils import (
    CLIPEncoderStackOp,
    ENCODER_PRE_INPUT_PATH,
    NUM_ENCODER_ONLY_DEPTHS,
    encoder_only_depth_name,
    load_encoder_pre_input,
    make_randns_like,
    RANDN_SEED_ENCODER,
    tensor_pcc,
)
from third_party.tt_forge_models.llava.pytorch.loader import ModelLoader

_MIN_PCC = 0.99


@pytest.fixture(scope="module")
def llava_encoder_layers():
    import torch

    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.eval()
    return model.model.vision_tower.vision_model.encoder.layers


@pytest.fixture(scope="module")
def encoder_pre_input_saved():
    if not ENCODER_PRE_INPUT_PATH.is_file():
        pytest.skip(f"Missing {ENCODER_PRE_INPUT_PATH}")
    return load_encoder_pre_input()


@pytest.fixture(scope="module")
def encoder_pre_input_randn(encoder_pre_input_saved):
    return make_randns_like(encoder_pre_input_saved, RANDN_SEED_ENCODER)


def _run_encoder_depth(layers, hidden_states, num_layers: int) -> float:
    op = CLIPEncoderStackOp(layers, num_layers)
    recorded: list[float] = []

    def comparator(device_output, golden_output, args, kwargs):
        recorded.append(tensor_pcc(device_output, golden_output))

    run_op_test(
        op,
        [hidden_states],
        comparison_config=ComparisonConfig(
            pcc=PccConfig(required_pcc=_MIN_PCC),
            assert_on_failure=False,
        ),
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(),
        custom_comparator=comparator,
    )
    assert recorded
    return recorded[0]


def _run_all_depths_report_failures(label: str, layers, hidden_states) -> None:
    lines = [f"{label} encoder-only PCC (threshold {_MIN_PCC}):"]
    failures: list[str] = []
    for n in NUM_ENCODER_ONLY_DEPTHS:
        name = encoder_only_depth_name(n)
        pcc = _run_encoder_depth(layers, hidden_states, n)
        lines.append(f"  {name}: pcc={pcc:.6f}")
        if pcc < _MIN_PCC:
            failures.append(f"{name} pcc={pcc:.6f} < {_MIN_PCC}")
    print("\n".join(lines))
    if failures:
        pytest.fail(
            f"{label}: {len(failures)} depth(s) below {_MIN_PCC}:\n"
            + "\n".join(failures)
            + "\n\n"
            + "\n".join(lines)
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_llava_encoder_only_saved_pre_input(llava_encoder_layers, encoder_pre_input_saved):
    _run_all_depths_report_failures(
        "encoder_only_saved",
        llava_encoder_layers,
        encoder_pre_input_saved,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_llava_encoder_only_randn_pre_input(llava_encoder_layers, encoder_pre_input_randn):
    _run_all_depths_report_failures(
        "encoder_only_randn",
        llava_encoder_layers,
        encoder_pre_input_randn,
    )

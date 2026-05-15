# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Progressive CLIPVisionTransformer op test: pixel_values through embeddings,
pre_layernorm, and encoder layers (exact modules from loaded LLaVA).

Runs every stack depth with required_pcc=0.99 for diagnosis; all stages run even
if some fail, then a single assertion reports every stage below threshold.

If PCC suddenly collapses or goes negative around a fixed depth, that is often
PyTorch Dynamo default recompile_limit (8), not real numerics — see conftest.py.

Requires: tests/torch/models/llava/fixtures/vision_pixel_values_bf16.pt
"""

from __future__ import annotations

import pytest
from infra import Framework, run_op_test
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig
from tests.torch.models.llava.encoder_utils import (
    CLIPVisionThroughEncoderOp,
    NUM_VISION_STACK_STAGES,
    VISION_PIXEL_VALUES_PATH,
    load_vision_pixel_values,
    make_randn_pixel_values,
    tensor_pcc,
    vision_stack_stage_name,
)
from third_party.tt_forge_models.llava.pytorch.loader import ModelLoader

_MIN_PCC = 0.99


@pytest.fixture(scope="module")
def llava_vision_tower_from_loader():
    import torch

    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.eval()
    vm = model.model.vision_tower.vision_model
    return {
        "embeddings": vm.embeddings,
        "pre_layernorm": vm.pre_layrnorm,
        "encoder_layers": vm.encoder.layers,
    }


@pytest.fixture(scope="module")
def vision_pixel_values():
    if not VISION_PIXEL_VALUES_PATH.is_file():
        pytest.skip(f"Missing {VISION_PIXEL_VALUES_PATH}")
    return load_vision_pixel_values()


@pytest.fixture(scope="module")
def vision_pixel_values_randn(vision_pixel_values):
    return make_randn_pixel_values(vision_pixel_values)


def _run_one_stage(fx, pixel_values, num_stages: int) -> float:
    op = CLIPVisionThroughEncoderOp(
        fx["embeddings"],
        fx["pre_layernorm"],
        fx["encoder_layers"],
        num_stages,
    )
    recorded: list[float] = []

    def comparator(device_output, golden_output, args, kwargs):
        recorded.append(tensor_pcc(device_output, golden_output))

    run_op_test(
        op,
        [pixel_values],
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


def _run_all_stages_report_failures(label: str, fx, pixel_values) -> None:
    lines = [f"{label} PCC vs golden (threshold {_MIN_PCC}):"]
    failures: list[str] = []
    for num_stages in NUM_VISION_STACK_STAGES:
        name = vision_stack_stage_name(num_stages)
        pcc = _run_one_stage(fx, pixel_values, num_stages)
        line = f"  {name}: pcc={pcc:.6f}"
        lines.append(line)
        if pcc < _MIN_PCC:
            failures.append(f"{name} pcc={pcc:.6f} < {_MIN_PCC}")
    print("\n".join(lines))
    if failures:
        pytest.fail(
            f"{label}: {len(failures)} stage(s) below {_MIN_PCC}:\n"
            + "\n".join(failures)
            + "\n\n"
            + "\n".join(lines)
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_llava_vision_through_encoder_saved_pixel_values(
    llava_vision_tower_from_loader, vision_pixel_values
):
    _run_all_stages_report_failures(
        "saved_pixel_values",
        llava_vision_tower_from_loader,
        vision_pixel_values,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_llava_vision_through_encoder_randn_pixel_values(
    llava_vision_tower_from_loader, vision_pixel_values_randn
):
    _run_all_stages_report_failures(
        "randn_pixel_values",
        llava_vision_tower_from_loader,
        vision_pixel_values_randn,
    )

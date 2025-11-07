# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_ttmlir_compilation

from third_party.tt_forge_models.whisper.pytorch import ModelLoader, ModelVariant

from .tester import WhisperTester

"""Run Whisper tests across all available variants."""


# ----- Fixtures -----

_FAILING_VARIANTS = (
    ModelVariant.WHISPER_LARGE,
    ModelVariant.WHISPER_LARGE_V3,
    ModelVariant.WHISPER_LARGE_V3_TURBO,
)


def _variant_param(v):
    """Create a pytest parameter for each ModelVariant with bringup_status and marks."""
    marks = []

    # Compute model info for specific variants
    model_info = ModelLoader.get_model_info(v)

    if v in _FAILING_VARIANTS:
        bringup_status = BringupStatus.FAILED_TTMLIR_COMPILATION
        marks.append(
            pytest.mark.xfail(
                reason=failed_ttmlir_compilation(
                    "RuntimeError: Not enough space to allocate 6710886400 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-xla/issues/1886"
                )
            )
        )
    else:
        bringup_status = BringupStatus.PASSED

    marks.extend(
        [
            pytest.mark.model_test,
            pytest.mark.record_test_properties(
                category=Category.MODEL_TEST,
                model_info=model_info,
                run_mode=RunMode.INFERENCE,
                bringup_status=bringup_status,
            ),
        ]
    )

    return pytest.param((v, bringup_status), marks=tuple(marks))


# Create parameter list + IDs
_WHISPER_PARAMS = [_variant_param(v) for v in list(ModelVariant)]
_WHISPER_IDS = [v.name.lower() for v in list(ModelVariant)]


@pytest.fixture(params=_WHISPER_PARAMS, ids=_WHISPER_IDS)
def inference_tester(request) -> WhisperTester:
    """Fixture that returns a WhisperTester configured for each model variant."""
    variant, bringup_status = request.param
    request.node.bringup_status = bringup_status
    return WhisperTester(variant)


@pytest.mark.nightly
def test_torch_whisper_inference(inference_tester: WhisperTester):
    inference_tester.test()


@pytest.mark.training
@pytest.fixture(params=_WHISPER_PARAMS, ids=_WHISPER_IDS)
@pytest.mark.skip(reason="Support for training not implemented")
def test_whisper_base_training(training_tester: WhisperTester):
    training_tester.test()

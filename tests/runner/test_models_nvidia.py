# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from infra.testers.single_chip.model.model_tester import RunMode

from tests.infra.evaluators.evaluator import Evaluator
from tests.infra.utilities.types import Framework
from tests.runner.requirements import RequirementsManager
from tests.runner.test_utils import (
    record_model_test_properties,
    update_test_metadata_for_exception,
)
from tests.runner.testers import DynamicTorchCudaModelTester
from tests.runner.utils import DynamicLoader, TorchDynamicLoader
from third_party.tt_forge_models.config import Parallelism

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


MODELS_ROOT_TORCH, test_entries_torch = TorchDynamicLoader.setup_test_discovery(
    PROJECT_ROOT
)
TEST_ENTRY_BY_ID = {
    DynamicLoader.generate_test_id(entry, MODELS_ROOT_TORCH): entry
    for entry in test_entries_torch
}


@dataclass
class NvidiaManifestRow:
    test_case_id: str
    display_name: str


class NvidiaModelInfo:
    """Minimal model-info surface for report-property compatibility."""

    def __init__(self, test_case_id: str, display_name: str):
        self.name = test_case_id
        self.model = display_name
        self.variant = display_name
        self.group = None
        self.task = "nvidia_validation"
        self.source = "ModelSource.HUGGING_FACE"
        self.framework = Framework.TORCH

    def to_report_dict(self):
        return {
            "name": self.name,
            "model": self.model,
            "variant": self.variant,
            "group": None,
            "task": self.task,
            "source": self.source,
            "framework": str(self.framework),
        }


def _load_nvidia_rows(path: str) -> list[NvidiaManifestRow]:
    obj = json.loads(Path(path).read_text())
    models = obj.get("models") or []
    rows = []
    for row in models:
        test_case_id = row.get("test_case_id")
        display_name = (
            row.get("display_name")
            or row.get("model_id")
            or row.get("pretrained_model_name")
            or test_case_id
        )
        if not test_case_id:
            continue
        if test_case_id not in TEST_ENTRY_BY_ID:
            continue
        rows.append(
            NvidiaManifestRow(
                test_case_id=test_case_id,
                display_name=display_name,
            )
        )
    return rows


def pytest_generate_tests(metafunc):
    if "nvidia_row" not in metafunc.fixturenames:
        return

    cohort_path = metafunc.config.getoption("--nvidia-cohort-json")
    if not cohort_path:
        metafunc.parametrize("nvidia_row", [], ids=[])
        return

    rows = _load_nvidia_rows(cohort_path)
    metafunc.parametrize(
        "nvidia_row",
        rows,
        ids=[row.test_case_id for row in rows],
    )


def _run_model_test_impl_nvidia(
    nvidia_row,
    record_property,
    test_metadata,
    request,
    captured_output_fixture,
):
    test_entry = TEST_ENTRY_BY_ID[nvidia_row.test_case_id]
    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader(variant=variant)
        model_info = NvidiaModelInfo(
            test_case_id=nvidia_row.test_case_id,
            display_name=nvidia_row.display_name,
        )

        succeeded = False
        comparison_result = None
        tester = None

        try:
            tester = DynamicTorchCudaModelTester(
                run_mode=RunMode.INFERENCE,
                loader=loader,
                comparison_config=test_metadata.to_comparison_config(),
                parallelism=Parallelism.SINGLE_DEVICE,
                test_metadata=test_metadata,
            )
            comparison_result = tester.test(request=request)
            succeeded = all(result.passed for result in comparison_result)
            Evaluator._assert_on_results(comparison_result)
        except Exception as e:
            try:
                captured = captured_output_fixture.readouterr()
                stdout, stderr = captured.out, captured.err
            except ValueError:
                stdout, stderr = None, None
            update_test_metadata_for_exception(
                test_metadata, e, stdout=stdout, stderr=stderr
            )
            raise
        finally:
            comparison_config = tester._comparison_config if tester else None
            model_size = getattr(tester, "_model_size", None) if tester else None
            record_model_test_properties(
                record_property,
                request,
                model_info=model_info,
                test_metadata=test_metadata,
                run_mode=RunMode.INFERENCE,
                parallelism=Parallelism.SINGLE_DEVICE,
                test_passed=succeeded,
                comparison_results=list(comparison_result) if comparison_result else [],
                comparison_config=comparison_config,
                model_size=model_size,
                weights_dtype="float32",
            )


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.nvidia
@pytest.mark.inference
@pytest.mark.single_device
def test_models_torch_nvidia(
    nvidia_row,
    record_property,
    test_metadata,
    request,
    captured_output_fixture,
):
    if nvidia_row is None:
        pytest.skip("No NVIDIA cohort rows selected")

    if not os.environ.get("CUDA_VISIBLE_DEVICES") and not torch.cuda.is_available():
        pytest.skip("CUDA not available for NVIDIA validation")

    _run_model_test_impl_nvidia(
        nvidia_row=nvidia_row,
        record_property=record_property,
        test_metadata=test_metadata,
        request=request,
        captured_output_fixture=captured_output_fixture,
    )

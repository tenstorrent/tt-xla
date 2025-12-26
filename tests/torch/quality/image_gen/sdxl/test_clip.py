# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra.evaluators import ComparisonConfig
from infra.testers.single_chip.model.model_tester import RunMode
from infra.testers.single_chip.quality.stable_diffusion_tester import (
    StableDiffusionTester,
)

from tests.utils import Category

from .data import CocoDataset
from .pipeline import SDXLConfig, SDXLPipeline

MODEL_INFO = {
    "name": "SDXL Pipeline",
    "task": "image_generation_quality",
    "height": 512,
    "width": 512,
    "num_samples": 10,
    "num_inference_steps": 50,
}


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.QUALITY_TEST,
    run_mode=RunMode.INFERENCE,
)
def test_clip_sdxl(request):
    dataset = CocoDataset()
    assert len(dataset.captions) == MODEL_INFO["num_samples"], (
        "Number of samples in the dataset does not match the pytest predefined "
        "number of samples. Consider updating the number of samples in the pytest properties."
    )

    pipeline_config = SDXLConfig(
        width=MODEL_INFO["width"],
        height=MODEL_INFO["height"],
    )

    comparison_config = ComparisonConfig()
    comparison_config.quality.min_clip_threshold = 25.0

    tester = StableDiffusionTester(
        pipeline_cls=SDXLPipeline,
        pipeline_config=pipeline_config,
        dataset=dataset,
        metric="clip",
        comparison_config=comparison_config,
        warmup=True,
        seed=42,
    )
    tester.test()

    # Serialize compilation artifacts if requested
    if request.config.getoption("--serialize", default=False):
        tester.serialize_compilation_artifacts(request.node.name)

# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.infra import RunMode
from tests.infra.testers.single_chip.quality import StableDiffusionTester
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
    tester = StableDiffusionTester(
        pipeline_cls=SDXLPipeline,
        pipeline_config=pipeline_config,
        dataset=dataset,
        metric="clip",
        min_threshold=25.0,
        warmup=True,
        seed=42,
    )
    tester.test()

    # Serialize compilation artifacts if requested
    if request.config.getoption("--serialize", default=False):
        tester.serialize_compilation_artifacts(request.node.name)

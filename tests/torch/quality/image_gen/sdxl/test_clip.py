import torch
import pytest
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from .pipeline import SDXLPipeline, SDXLConfig
from .data import CocoDataset
from .metrics import CLIPMetric
from tests.utils import Category
from tests.infra import RunMode

MODEL_INFO = {
        "name": "SDXL Pipeline",
        "task": "image_generation_quality",
        "height": 512,
        "width": 512,
        "num_samples": 10,
        "num_inference_steps": 50
    }

@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.QUALITY_TEST,
    run_mode=RunMode.INFERENCE
    )
def test_clip_sdxl():
    xr.set_device_type("TT")

    pipeline = SDXLPipeline(config=SDXLConfig(width=MODEL_INFO["width"], height=MODEL_INFO["height"]))
    pipeline.setup(warmup=True)
    dataset = CocoDataset()
    assert len(dataset.captions) == MODEL_INFO["num_samples"], "Number of samples in the dataset does not match the pytest predefined number of samples. Consider updating the number of samples in the pytest properties."

    images = []
    for caption in dataset.captions:
        img = pipeline.generate(caption, seed=42)
        images.append(img)
    
    images = torch.cat(images, dim=0)
    clip_metric = CLIPMetric()

    _, clip_min = clip_metric.compute(images, dataset.captions)
    assert clip_min > 25.0, "CLIP score regression detected"

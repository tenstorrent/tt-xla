# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Sequence

import jax
import pytest
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import FlaxViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests
from utils import record_model_test_properties, runtime_fail



class FlaxViTForImageClassificationTester(ModelTester):
    """Tester for Vision Transformer model on an image classification task using Flax."""

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxViTForImageClassification:
        return FlaxViTForImageClassification.from_pretrained(self._model_name)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        processor = ViTImageProcessor.from_pretrained(self._model_name)
        inputs = processor(images=image, return_tensors="jax")
        return inputs["pixel_values"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "pixel_values": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----

@pytest.fixture
def inference_tester(request) -> FlaxViTForImageClassificationTester:
    return FlaxViTForImageClassificationTester(request.param)


@pytest.fixture
def training_tester(request) -> FlaxViTForImageClassificationTester:
    return FlaxViTForImageClassificationTester(request.param, RunMode.TRAINING)


# ----- Tests -----

@pytest.mark.nightly
@pytest.mark.parametrize(
    "inference_tester", 
   [   
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-384",
    "google/vit-base-patch16-224",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch16-224",
    "google/vit-large-patch16-384",
    "google/vit-large-patch32-224-in21k",
    "google/vit-large-patch32-384",
    "google/vit-huge-patch14-224-in21k", 
   ],
   indirect=True, 
   ids=lambda val: val)
@pytest.mark.xfail(
    reason=runtime_fail(
        "Out of memory while performing convolution."
        "(https://github.com/tenstorrent/tt-xla/issues/308)"
    )
)
def test_vit_inference(
    inference_tester: FlaxViTForImageClassificationTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, inference_tester._model_name)
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.parametrize(
    "training_tester", 
    [   
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-384",
    "google/vit-base-patch16-224",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch16-224",
    "google/vit-large-patch16-384",
    "google/vit-large-patch32-224-in21k",
    "google/vit-large-patch32-384",
    "google/vit-huge-patch14-224-in21k", 
   ], 
    indirect=True, 
    ids=lambda val: val)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_training(
    training_tester: FlaxViTForImageClassificationTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, training_tester._model_name)
    training_tester.test()

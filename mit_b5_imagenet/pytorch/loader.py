# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiT-B5 ImageNet model loader implementation for image segmentation.

Uses the segmentation-models-pytorch (smp) library to load a segmentation
model with MiT-B5 encoder pretrained on ImageNet from smp-hub/mit_b5.imagenet.
"""
import torch
from PIL import Image
from typing import Optional
from torchvision import transforms
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import VisionPreprocessor


class ModelVariant(StrEnum):
    """Available MiT-B5 ImageNet model variants."""

    MIT_B5_IMAGENET = "Mit_B5_ImageNet"


class ModelLoader(ForgeModel):
    """MiT-B5 ImageNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.MIT_B5_IMAGENET: ModelConfig(
            pretrained_model_name="smp-hub/mit_b5.imagenet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIT_B5_IMAGENET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiT-B5 ImageNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the MiT-B5 segmentation model from segmentation-models-pytorch."""
        import segmentation_models_pytorch as smp

        model = smp.from_pretrained(self._variant_config.pretrained_model_name)
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) for MiT-B5.

        Uses standard ImageNet preprocessing: resize to 224x224, normalize
        with ImageNet mean/std.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                return preprocess(img)

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name=model_name,
                custom_preprocess_fn=custom_preprocess_fn,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

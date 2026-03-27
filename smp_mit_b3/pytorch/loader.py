# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SMP MiT-B3 (Mix Transformer B3) model loader implementation using segmentation-models-pytorch.
"""
import torch
from typing import Optional
from PIL import Image
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
    """Available SMP MiT-B3 model variants."""

    MIT_B3_IMAGENET = "Mit_B3_Imagenet"


class ModelLoader(ForgeModel):
    """SMP MiT-B3 model loader implementation for semantic segmentation."""

    _VARIANTS = {
        ModelVariant.MIT_B3_IMAGENET: ModelConfig(
            pretrained_model_name="smp-hub/mit_b3.imagenet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIT_B3_IMAGENET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SMP-MiT-B3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SMP MiT-B3 U-Net model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The SMP U-Net model with MiT-B3 encoder.
        """
        import segmentation_models_pytorch as smp

        model = smp.Unet("mit_b3", encoder_weights="imagenet")
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def _create_custom_preprocess_fn(self):
        """Create preprocessing function using SMP encoder params."""
        import segmentation_models_pytorch as smp

        params = smp.encoders.get_preprocessing_params("mit_b3")
        std = torch.tensor(params["std"]).view(1, 3, 1, 1)
        mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

        def preprocess_fn(image: Image.Image) -> torch.Tensor:
            img = image.convert("RGB")
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)

            _, _, h, w = img_tensor.shape
            output_stride = 32
            new_h = ((h - 1) // output_stride + 1) * output_stride
            new_w = ((w - 1) // output_stride + 1) * output_stride

            if h != new_h or w != new_w:
                pad_h = new_h - h
                pad_w = new_w - w
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
                )

            return (img_tensor - mean) / std

        return preprocess_fn

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size.
            image: PIL Image, URL string, tensor, list, or None (uses default).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            custom_preprocess_fn = self._create_custom_preprocess_fn()

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name="smp-mit-b3",
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
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

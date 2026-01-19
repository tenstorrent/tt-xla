# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RGBM model loader implementation for image segmentation
"""
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available RMBG model variants."""

    RMBG_2_0 = "2_0"


class ModelLoader(ForgeModel):
    """RMBG model loader implementation for image segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.RMBG_2_0: ModelConfig(
            pretrained_model_name="briaai/RMBG-2.0",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RMBG_2_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="RMBG",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        """Setup image transforms for preprocessing.

        Returns:
            torchvision.transforms.Compose: The image transform pipeline
        """
        # Image size and transforms
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return self.transform_image

    def load_model(self, dtype_override=None):
        """Load and return the RMBG model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The RMBG model instance for image segmentation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Load pre-trained model from HuggingFace
        model = AutoModelForImageSegmentation.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )

        # Set matmul precision
        torch.set_float32_matmul_precision(["high", "highest"][0])

        # Setup transforms
        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RMBG model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor (preprocessed image) that can be fed to the model.
        """
        # Ensure transforms are setup
        if self.transform_image is None:
            self._setup_transforms()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = Image.open(str(image_file))

        inputs = self.transform_image(self.image).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        # Add batch dimension
        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs, save_image=False, output_path="no_bg_image.png"):
        """Helper method to decode model outputs into segmentation mask and optionally save result.

        Args:
            outputs: Model output from a forward pass
            save_image: Whether to save the result image with transparent background
            output_path: Path to save the output image

        Returns:
            str: Information about the segmentation result
        """
        if self.image is None:
            return "Error: No input image loaded for processing"

        # Process predictions
        predictions = outputs[-1].sigmoid()
        pred = predictions[0].squeeze()
        pred = pred.to(torch.float32)

        # Convert to PIL image and resize to match original
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(self.image.size)

        result_info = f"""
        RMBG Segmentation Output:
          - Original image size: {self.image.size}
          - Prediction shape: {predictions.shape}
          - Mask size: {mask.size}
        """

        # Optionally save image with transparent background
        if save_image:
            result_image = self.image.copy()
            result_image.putalpha(mask)
            result_image.save(output_path)
            result_info += f"  - Saved result to: {output_path}"

        return result_info

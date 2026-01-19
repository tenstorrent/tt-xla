# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MGP-STR model loader implementation
"""

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from PIL import Image
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="mgp_str_base",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    """Loads MGP-STR model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "alibaba-damo/mgp-str-base"

    def load_model(self, dtype_override=None):
        """Load pretrained MGP-STR model."""

        model = MgpstrForSceneTextRecognition.from_pretrained(self.model_name)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for MGP-STR model"""

        # Get the Image
        image_file = get_file("https://i.postimg.cc/ZKwLg2Gw/367-14.png")
        image = Image.open(image_file).convert("RGB")

        # Preprocess image
        self.processor = MgpstrProcessor.from_pretrained(self.model_name)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        # Add batch dimension
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, co_out):
        """Helper method to decode model outputs into human-readable text.

        Args:
            co_out: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        output = (co_out[0], co_out[1], co_out[2])

        generated_text = self.processor.batch_decode(output)["generated_text"]
        print(f"Generated text: {generated_text}")
        return generated_text

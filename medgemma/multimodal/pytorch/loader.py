# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedGemma model loader implementation for multimodal modeling.
"""

from typing import Optional

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available MedGemma multimodal model variants."""

    MEDGEMMA_4B_PT = "google/medgemma-4b-pt"


class ModelLoader(ForgeModel):
    """MedGemma model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEDGEMMA_4B_PT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MEDGEMMA_4B_PT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDGEMMA_4B_PT

    sample_text = "Describe this medical image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="medgemma_multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name, **kwargs)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MedGemma multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MedGemma model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the MedGemma multimodal model.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names) where mesh_shape is (batch_dim, model_dim)
                   and mesh_axis_names are ("batch", "model").
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

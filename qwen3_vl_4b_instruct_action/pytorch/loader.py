# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL-4B-Instruct-action model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3-VL-4B-Instruct-action model variants."""

    QWEN3_VL_4B_INSTRUCT_ACTION = "4b_instruct_action"


class ModelLoader(ForgeModel):
    """Qwen3-VL-4B-Instruct-action model loader for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION: LLMModelConfig(
            pretrained_model_name="229nagibator229/Qwen3-VL-4B-Instruct-action",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="qwen3_vl_4b_instruct_action",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-VL-4B-Instruct-action model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, dtype="auto", device_map="auto", **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

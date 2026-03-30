# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen 2.5 VL GGUF model variants for image to text."""

    MRADERMACHER_QWEN_2_5_VL_32B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "mradermacher_32B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_QWEN_2_5_VL_32B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen2.5-VL-32B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.MRADERMACHER_QWEN_2_5_VL_32B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF
    )

    _GGUF_FILES = {
        ModelVariant.MRADERMACHER_QWEN_2_5_VL_32B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Qwen2.5-VL-32B-Instruct-abliterated.Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

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
        return ModelInfo(
            model="Qwen 2.5 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 VL GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 2.5 VL GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 2.5 VL GGUF model.

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
                        "image": self.sample_image,
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

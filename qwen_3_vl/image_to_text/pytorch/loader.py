# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
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
    """Available Qwen 3 model variants for image to text."""

    QWEN_3_VL_2B_INSTRUCT = "2b_instruct"
    QWEN_3_VL_2B_THINKING = "2b_thinking"
    QWEN_3_VL_4B_INSTRUCT = "4b_instruct"
    QWEN_3_VL_4B_THINKING = "4b_thinking"
    QWEN_3_VL_8B_INSTRUCT = "8b_instruct"
    QWEN_3_VL_8B_INSTRUCT_FP8 = "8b_instruct_fp8"
    QWEN_3_VL_30B_A3B_INSTRUCT = "30b_a3b_instruct"
    QWEN_3_VL_30B_A3B_THINKING_FP8 = "30b_a3b_thinking_fp8"
    QWEN_3_VL_8B_INSTRUCT_UNSLOTH_BNB_4BIT = "8b_instruct_unsloth_bnb_4bit"
    QWEN_3_VL_32B_INSTRUCT = "32b_instruct"
    HUIHUI_QWEN_3_VL_4B_INSTRUCT_ABLITERATED = "huihui_4b_instruct_abliterated"


class ModelLoader(ForgeModel):
    """Qwen 3 model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_VL_2B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-2B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_2B_THINKING: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-2B-Thinking",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_4B_THINKING: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Thinking",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_8B_INSTRUCT_UNSLOTH_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_THINKING_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
            max_length=128,
        ),
        ModelVariant.HUIHUI_QWEN_3_VL_4B_INSTRUCT_ABLITERATED: LLMModelConfig(
            pretrained_model_name="huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
            max_length=128,
        ),
    }

    # Variants that use the MoE architecture
    _MOE_VARIANTS = {
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT,
        ModelVariant.QWEN_3_VL_30B_A3B_THINKING_FP8,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_2B_INSTRUCT

    # Shared configuration parameters
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        group = (
            ModelGroup.VULCAN
            if variant
            in (
                ModelVariant.QWEN_3_VL_8B_INSTRUCT,
                ModelVariant.QWEN_3_VL_8B_INSTRUCT_FP8,
                ModelVariant.QWEN_3_VL_8B_INSTRUCT_UNSLOTH_BNB_4BIT,
                ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT,
                ModelVariant.QWEN_3_VL_30B_A3B_THINKING_FP8,
                ModelVariant.QWEN_3_VL_32B_INSTRUCT,
                ModelVariant.HUIHUI_QWEN_3_VL_4B_INSTRUCT_ABLITERATED,
            )
            else ModelGroup.RED
        )
        return ModelInfo(
            model="qwen_v3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3 model instance for image to text.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # AWQ variant loads with device_map="cpu" to keep quantized weights on CPU
        if self._variant == ModelVariant.QWEN_3_VL_4B_INSTRUCT_AWQ:
            model_kwargs["device_map"] = "cpu"
        else:
            model_kwargs["dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if self._variant in self._MOE_VARIANTS
            else Qwen3VLForConditionalGeneration
        )
        model = model_cls.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3 model with this instance's variant settings.

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

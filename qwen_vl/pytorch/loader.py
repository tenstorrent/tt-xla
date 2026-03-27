# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-VL model loader implementation for vision-language tasks.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen-VL model variants."""

    QWEN_VL = "qwen_vl"
    QWEN_VL_CHAT = "qwen_vl_chat"


class ModelLoader(ForgeModel):
    """Qwen-VL model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_VL: ModelConfig(
            pretrained_model_name="Qwen/Qwen-VL",
        ),
        ModelVariant.QWEN_VL_CHAT: ModelConfig(
            pretrained_model_name="Qwen/Qwen-VL-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_VL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        query = self.tokenizer.from_list_format(
            [
                {
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"text": "Generate the caption in English with grounding:"},
            ]
        )

        inputs = self.tokenizer(query, return_tensors="pt")
        return inputs

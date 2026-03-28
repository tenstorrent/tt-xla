# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL Reranker model loader implementation for multimodal reranking tasks.
"""

import torch
from transformers import AutoProcessor, Qwen3VLForSequenceClassification
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


class ModelVariant(StrEnum):
    """Available Qwen 3 VL Reranker model variants."""

    QWEN_3_VL_RERANKER_8B = "reranker_8b"


class ModelLoader(ForgeModel):
    """Qwen 3 VL Reranker model loader for multimodal reranking tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_RERANKER_8B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-Reranker-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_RERANKER_8B

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL Reranker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen3VLForSequenceClassification.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {
                        "type": "text",
                        "text": "A woman playing with her dog on a beach at sunset.",
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        scores = torch.sigmoid(logits).squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

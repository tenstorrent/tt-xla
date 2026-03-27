# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 MLC model loader implementation for causal language modeling.
"""
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
    """Available Qwen 2.5 MLC model variants for causal language modeling."""

    QWEN_2_5_0_5B_INSTRUCT_Q4F16_1_MLC = "0.5B_Instruct_q4f16_1_MLC"


class ModelLoader(ForgeModel):
    """Qwen 2.5 MLC model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_0_5B_INSTRUCT_Q4F16_1_MLC: LLMModelConfig(
            pretrained_model_name="mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_0_5B_INSTRUCT_Q4F16_1_MLC

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.engine = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 2.5 MLC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from mlc_llm import MLCEngine

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_path = f"HF://{pretrained_model_name}"

        self.engine = MLCEngine(model_path)
        return self.engine

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        return {
            "messages": messages,
            "model": self._variant_config.pretrained_model_name,
        }

    def decode_output(self, outputs):
        if hasattr(outputs, "choices") and outputs.choices:
            return outputs.choices[0].message.content
        return str(outputs)

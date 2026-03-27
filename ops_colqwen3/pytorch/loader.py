# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ops-ColQwen3 model loader implementation for multimodal document retrieval embedding tasks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
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
    """Available Ops-ColQwen3 model variants."""

    OPS_COLQWEN3_4B = "4B"


class ModelLoader(ForgeModel):
    """Ops-ColQwen3 model loader for multimodal document retrieval embedding tasks."""

    _VARIANTS = {
        ModelVariant.OPS_COLQWEN3_4B: ModelConfig(
            pretrained_model_name="OpenSearch-AI/Ops-Colqwen3-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPS_COLQWEN3_4B

    sample_queries = [
        "What is the revenue growth for Q3 2024?",
    ]
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ops-ColQwen3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        # Format a text query input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.sample_queries[0]},
                ],
            }
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
        # Extract hidden states from model output
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        # Normalize the multi-vector embeddings
        embeddings = F.normalize(hidden_states, p=2, dim=-1)

        return embeddings.tolist()

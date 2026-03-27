# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL Embedding model loader implementation for multimodal embedding tasks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
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
from .src.utils import last_token_pool


class ModelVariant(StrEnum):
    """Available Qwen 3 VL Embedding model variants."""

    QWEN_3_VL_EMBEDDING_2B = "Embedding_2B"
    QWEN_3_VL_EMBEDDING_8B = "Embedding_8B"


class ModelLoader(ForgeModel):
    """Qwen 3 VL Embedding model loader for multimodal embedding tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_EMBEDDING_2B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-Embedding-2B",
        ),
        ModelVariant.QWEN_3_VL_EMBEDDING_8B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-Embedding-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_EMBEDDING_2B

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    sample_queries = [
        {"text": "A woman and a dog playing on the beach"},
    ]
    sample_documents = [
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        },
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3 VL Embedding",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
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

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def _format_input(self, item, instruction="Represent the user's input."):
        """Format an input item into a chat conversation for the processor."""
        content = []
        if "image" in item:
            content.append({"type": "image", "image": item["image"]})
        if "text" in item:
            content.append({"type": "text", "text": item["text"]})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]
        return messages

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        all_items = self.sample_queries + self.sample_documents
        messages_list = [self._format_input(item) for item in all_items]

        # Process the first item as a representative input
        messages = messages_list[0]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        # Extract last hidden state
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        # Pool embeddings using last token
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            embeddings = last_token_pool(hidden_states, attention_mask)
        else:
            embeddings = hidden_states[:, -1]

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

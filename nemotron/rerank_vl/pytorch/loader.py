# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron Rerank VL model loader implementation for multimodal document reranking.
"""

import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor
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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Nemotron Rerank VL model variants."""

    LLAMA_NEMOTRON_RERANK_VL_1B_V2 = "Llama_Nemotron_Rerank_VL_1B_V2"


class ModelLoader(ForgeModel):
    """Nemotron Rerank VL model loader for multimodal document reranking."""

    _VARIANTS = {
        ModelVariant.LLAMA_NEMOTRON_RERANK_VL_1B_V2: ModelConfig(
            pretrained_model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_NEMOTRON_RERANK_VL_1B_V2

    sample_query = "How is AI improving robotics?"
    sample_doc_text = (
        "Artificial intelligence is revolutionizing robotics by enabling machines "
        "to perceive, learn, and adapt to their environments autonomously."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Nemotron Rerank VL model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NemotronRerankVL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=2048,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nemotron Rerank VL model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Nemotron Rerank VL model."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        doc_image = Image.open(image_file)

        examples = [
            {
                "question": self.sample_query,
                "doc_text": self.sample_doc_text,
                "doc_image": doc_image,
            }
        ]

        if batch_size > 1:
            examples = examples * batch_size

        batch_dict = self.processor.process_queries_documents_crossencoder(examples)

        if dtype_override is not None:
            batch_dict = {
                k: v.to(dtype_override)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v
                for k, v in batch_dict.items()
            }

        return batch_dict

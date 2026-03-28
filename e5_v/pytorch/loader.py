# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
E5-V multimodal embedding model loader.

E5-V produces universal embeddings from both images and text using a
LLaVA-NeXT backbone. Embeddings are extracted from the last hidden state
of the last token and L2-normalized.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
from ...tools.utils import get_file


LLAMA3_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"


class ModelVariant(StrEnum):
    """Available E5-V model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """E5-V multimodal embedding model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="royokong/e5-v",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="E5-V",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the E5-V model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaNextForConditionalGeneration.from_pretrained(
            str(model_name), torch_dtype=torch.float16, **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for E5-V.

        Prepares an image input using the LLaMA 3 prompt template expected
        by E5-V for extracting image embeddings.
        """
        if self.processor is None:
            self._load_processor()

        img_prompt = LLAMA3_TEMPLATE.format(
            "<image>\nSummary above image in one word: "
        )

        image_file = get_file(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/360px-American_Eskimo_Dog.jpg"
        )
        image = Image.open(image_file)

        inputs = self.processor(
            images=image, text=img_prompt, return_tensors="pt", padding=True
        )

        if dtype_override:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        """Extract and normalize the embedding from model output.

        E5-V embeddings are the last hidden state at the last token position,
        L2-normalized.
        """
        hidden_states = output.hidden_states[-1]
        emb = hidden_states[:, -1, :]
        return F.normalize(emb, dim=-1)

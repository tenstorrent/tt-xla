# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moondream2 model loader implementation for vision-language tasks.
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available Moondream2 model variants."""

    MOONDREAM2 = "2B"


class ModelLoader(ForgeModel):
    """Moondream2 model loader for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.MOONDREAM2: ModelConfig(
            pretrained_model_name="vikhyatk/moondream2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOONDREAM2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Moondream2 model loader."""
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moondream2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Moondream2 model instance."""
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision="2025-06-21", trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True, "revision": "2025-06-21"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return input tensors for Moondream2."""
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        # Load a sample image
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Use the model's built-in encoding to get input embeddings
        enc_image = self.model.encode_image(image)
        inputs_embeds = self.model.input_embeds("Describe this image.", enc_image)

        return inputs_embeds

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
nanoLLaVA model loader implementation for multimodal conditional generation.
"""

import torch
from PIL import Image
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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available nanoLLaVA model variants."""

    NANOLLAVA = "nanoLLaVA"


class ModelLoader(ForgeModel):
    """nanoLLaVA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.NANOLLAVA: ModelConfig(
            pretrained_model_name="qnguyen3/nanoLLaVA",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANOLLAVA

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize nanoLLaVA model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="nanoLLaVA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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
        """Load and return the nanoLLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for nanoLLaVA."""
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {"role": "user", "content": f"<image>\n{self.sample_text}"},
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and insert image placeholder token (-200) per model convention
        text_chunks = text_prompt.split("<image>")
        input_ids = (
            self.tokenizer(text_chunks[0], return_tensors="pt").input_ids,
            torch.tensor([[-200]], dtype=torch.long),
            self.tokenizer(
                text_chunks[1], return_tensors="pt", add_special_tokens=False
            ).input_ids,
        )
        input_ids = torch.cat(input_ids, dim=1)

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        # Load model temporarily to process the image
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        images = model.process_images([image], model.config).to(dtype=torch.float16)
        del model

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            images = cast_input_to_type(images, dtype_override)

        return {
            "input_ids": input_ids,
            "images": images,
        }

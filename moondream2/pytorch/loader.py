# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moondream2 model loader implementation for vision-language tasks.
"""

from typing import Optional

import torch
import torch.nn as nn
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


class Moondream2TextDecoderWrapper(nn.Module):
    """Wrapper around Moondream2's text decoder for standard forward pass."""

    def __init__(self, moondream_model):
        super().__init__()
        self.moondream = moondream_model.model
        self.config = moondream_model.model.config

    def forward(self, inputs_embeds):
        self.moondream._setup_caches()
        seq_len = inputs_embeds.size(1)
        attn_mask = self.moondream.attn_mask[:, :, :seq_len, :]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
        hidden = self.moondream._prefill(inputs_embeds, attn_mask, pos_ids, None)
        return hidden


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
        self.raw_model = None

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
        """Load and return the Moondream2 model wrapped for standard forward pass."""
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

        self.raw_model = model
        return Moondream2TextDecoderWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return input embeddings for Moondream2."""
        if self.raw_model is None:
            self.load_model(dtype_override=dtype_override)

        mm = self.raw_model.model

        # Load a sample image
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Run vision encoder to get image embeddings
        with torch.no_grad():
            img_emb = mm._run_vision_encoder(image)
            bos_emb = torch.nn.functional.embedding(
                torch.tensor([[mm.config.tokenizer.bos_id]], device=mm.device),
                mm.text.wte,
            )
            # Combine BOS token embedding with image embeddings
            inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)

        return inputs_embeds.clone().detach()

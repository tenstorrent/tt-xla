# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLAP-IPA phone encoder model loader implementation for embedding generation.
"""
import torch
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
    """Available CLAP-IPA phone encoder model variants."""

    TINY_PHONE = "Tiny_Phone"


class ModelLoader(ForgeModel):
    """CLAP-IPA phone encoder model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.TINY_PHONE: ModelConfig(
            pretrained_model_name="anyspeech/clap-ipa-tiny-phone",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_PHONE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CLAP_IPA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import BertModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        # The model uses a phoneme vocabulary (vocab_size=450, max_length=514).
        # No tokenizer is provided, so we generate synthetic phone token IDs.
        seq_length = 32
        vocab_size = 450
        input_ids = torch.randint(1, vocab_size, (1, seq_length))
        attention_mask = torch.ones(1, seq_length, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

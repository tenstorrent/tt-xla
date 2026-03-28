# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NTv3 Post-Trained model loader implementation for masked language modeling on DNA sequences.
"""
from transformers import AutoTokenizer, AutoModel
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
    """Available NTv3 Post-Trained model variants."""

    NTV3_650M_POST = "InstaDeepAI/NTv3_650M_post"


class ModelLoader(ForgeModel):
    """NTv3 Post-Trained model loader implementation for masked language modeling on DNA sequences."""

    _VARIANTS = {
        ModelVariant.NTV3_650M_POST: ModelConfig(
            pretrained_model_name="InstaDeepAI/NTv3_650M_post",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NTV3_650M_POST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NTv3PostTrained",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
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
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use a DNA sequence with <mask> token for masked LM task
        masked_sequence = "ACCTGA<mask>TTCTGAGTC"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return inputs

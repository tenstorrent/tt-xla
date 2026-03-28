# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
IndicTrans2 model loader implementation for text translation.
"""

import torch
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
    """Available IndicTrans2 model variants."""

    EN_INDIC_DIST_200M = "En_Indic_Dist_200M"


class ModelLoader(ForgeModel):
    """IndicTrans2 model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.EN_INDIC_DIST_200M: LLMModelConfig(
            pretrained_model_name="ai4bharat/indictrans2-en-indic-dist-200M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EN_INDIC_DIST_200M

    sample_text = "Hello, how are you today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="IndicTrans2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the IndicTrans2 model instance."""
        from transformers import AutoModelForSeq2SeqLM

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"trust_remote_code": True, "return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the IndicTrans2 model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]])
        inputs["decoder_input_ids"] = decoder_input_ids

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

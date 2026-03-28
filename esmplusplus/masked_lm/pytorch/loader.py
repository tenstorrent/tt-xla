# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESM++ model loader implementation for masked language modeling on protein sequences.
"""
import torch
from transformers import AutoModelForMaskedLM
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
    """Available ESM++ model variants."""

    ESMPLUSPLUS_SMALL = "Synthyra/ESMplusplus_small"


class ModelLoader(ForgeModel):
    """ESM++ model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.ESMPLUSPLUS_SMALL: ModelConfig(
            pretrained_model_name="Synthyra/ESMplusplus_small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESMPLUSPLUS_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESMplusplus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, model):
        self.tokenizer = model.tokenizer
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        if self.tokenizer is None:
            self._load_tokenizer(model)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            raise RuntimeError("load_model() must be called before load_inputs()")

        masked_sequence = "MGSSHHHHHHSSGLVPRGSHM<mask>GSSHHHHHHSSGLVPRGSHM"

        inputs = self.tokenizer(
            masked_sequence,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            raise RuntimeError("load_model() must be called before decode_output()")

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens

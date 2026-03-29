# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ClimateBERT model loader implementation for masked language modeling.
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM

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
    """Available ClimateBERT model variants."""

    DISTILROBERTA_BASE_CLIMATE_F = "DistilRoBERTa_Base_Climate_F"


class ModelLoader(ForgeModel):
    """ClimateBERT model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.DISTILROBERTA_BASE_CLIMATE_F: LLMModelConfig(
            pretrained_model_name="climatebert/distilroberta-base-climate-f",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILROBERTA_BASE_CLIMATE_F

    sample_text = (
        "Climate change is expected to cause significant <mask> in global temperatures."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ClimateBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs):
        if self._tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        inputs = self.load_inputs()

        mask_token_index = (inputs["input_ids"] == self._tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        output = self._tokenizer.decode(predicted_token_id)

        return f"Output: {output}"

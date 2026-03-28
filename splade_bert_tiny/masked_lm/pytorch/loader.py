# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SPLADE BERT Tiny model loader implementation for masked language modeling."""

from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    """Available SPLADE BERT Tiny model variants."""

    SPLADE_BERT_TINY_NQ = "splade-bert-tiny-nq"


class ModelLoader(ForgeModel):
    """SPLADE BERT Tiny model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.SPLADE_BERT_TINY_NQ: LLMModelConfig(
            pretrained_model_name="sparse-encoder-testing/splade-bert-tiny-nq",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPLADE_BERT_TINY_NQ

    sample_text = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""

        return ModelInfo(
            model="SPLADE BERT Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SPLADE BERT Tiny model instance."""

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self._tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for masked language modeling."""

        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self._tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self._tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)

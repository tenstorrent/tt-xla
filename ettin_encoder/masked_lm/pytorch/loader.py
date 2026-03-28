# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ettin Encoder model loader implementation for masked language modeling.
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Ettin Encoder model variants for masked language modeling."""

    ETTIN_ENCODER_68M = "68M"
    ETTIN_ENCODER_150M = "150M"
    ETTIN_ENCODER_280M = "280M"
    ETTIN_ENCODER_580M = "580M"


class ModelLoader(ForgeModel):
    """Ettin Encoder model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.ETTIN_ENCODER_68M: LLMModelConfig(
            pretrained_model_name="jhu-clsp/ettin-encoder-68m",
            max_length=128,
        ),
        ModelVariant.ETTIN_ENCODER_150M: LLMModelConfig(
            pretrained_model_name="jhu-clsp/ettin-encoder-150m",
            max_length=128,
        ),
        ModelVariant.ETTIN_ENCODER_280M: LLMModelConfig(
            pretrained_model_name="jhu-clsp/ettin-encoder-280m",
            max_length=128,
        ),
        ModelVariant.ETTIN_ENCODER_580M: LLMModelConfig(
            pretrained_model_name="jhu-clsp/ettin-encoder-580m",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ETTIN_ENCODER_68M

    def __init__(self, variant=None):
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "The capital of France is [MASK]."
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ettin Encoder",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
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
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config

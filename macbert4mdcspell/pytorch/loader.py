# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MacBERT4mdcspell model loader implementation for Chinese spelling correction.
"""

from transformers import BertForMaskedLM, BertTokenizer
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
    """Available MacBERT4mdcspell model variants."""

    V1 = "Macropodus/macbert4mdcspell_v1"


class ModelLoader(ForgeModel):
    """MacBERT4mdcspell model loader for Chinese spelling correction."""

    _VARIANTS = {
        ModelVariant.V1: LLMModelConfig(
            pretrained_model_name="Macropodus/macbert4mdcspell_v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant=None):
        super().__init__(variant)
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "我今天去了北京天按门。"
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="MacBERT4mdcspell",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
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
        """Decode the model output for Chinese spelling correction."""
        inputs = self.load_inputs()
        logits = co_out[0]
        predicted_ids = logits[0].argmax(dim=-1)
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        corrected_ids = input_ids.clone()
        corrected_ids[attention_mask == 1] = predicted_ids[attention_mask == 1]
        corrected_text = self.tokenizer.decode(
            corrected_ids[attention_mask == 1], skip_special_tokens=True
        )
        print(f"Original: {self.sample_text}")
        print(f"Corrected: {corrected_text}")

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NLLB-200 3.3B IWSLT26-IT model loader implementation for text translation.
"""
import torch
from typing import Optional

from peft import PeftModel

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
    """Available NLLB-200 3.3B IWSLT26-IT model variants."""

    NLLB_200_3_3B_IWSLT26_IT = "NLLB_200_3.3B_IWSLT26_IT"


class ModelLoader(ForgeModel):
    """NLLB-200 3.3B IWSLT26-IT model loader for text translation tasks."""

    _VARIANTS = {
        ModelVariant.NLLB_200_3_3B_IWSLT26_IT: LLMModelConfig(
            pretrained_model_name="jorirsan/nllb-200-3.3B-iwslt26-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NLLB_200_3_3B_IWSLT26_IT

    BASE_MODEL_NAME = "facebook/nllb-200-3.3B"

    sample_text = "Hello, how are you today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NLLB-200-3.3B-IWSLT26-IT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSeq2SeqLM

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        self.tokenizer.src_lang = "eng_Latn"

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        # Use the target language BOS token to start decoding into Italian.
        target_lang_id = self.tokenizer.convert_tokens_to_ids("ita_Latn")
        inputs["decoder_input_ids"] = torch.tensor([[target_lang_id]])

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MarianMT model loader implementation for text classification.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
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
    """Available MarianMT model variants."""

    OPUS_MT_NL_EN = "Opus_Mt_Nl_En"


class ModelLoader(ForgeModel):
    """MarianMT model loader implementation for text classification."""

    _VARIANTS = {
        ModelVariant.OPUS_MT_NL_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-nl-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPUS_MT_NL_EN

    _SAMPLE_TEXTS = {
        ModelVariant.OPUS_MT_NL_EN: "Mijn vrienden zijn cool maar ze eten te veel koolhydraten.",
    }

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        group = ModelGroup.VULCAN
        return ModelInfo(
            model="MarianMT",
            variant=variant_name,
            group=group,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MarianMT model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MarianMT model instance.
        """
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MarianMTModel.from_pretrained(self.model_name, **model_kwargs)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MarianMT model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant)
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Seq2seq models require decoder_input_ids for forward pass
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for translation.

        Args:
            co_out: Model output logits
        """
        translated_tokens = co_out[0].argmax(-1)
        translation = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        print(f"Translation: {translation}")

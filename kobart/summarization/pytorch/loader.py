# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoBART model loader implementation for summarization.
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
    """Available KoBART model variants for summarization."""

    V3 = "V3"


class ModelLoader(ForgeModel):
    """KoBART model loader implementation for summarization."""

    _VARIANTS = {
        ModelVariant.V3: LLMModelConfig(
            pretrained_model_name="EbanLee/kobart-summary-v3",
            max_length=1026,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V3

    sample_text = (
        "서울특별시는 대한민국의 수도이자 최대 도시이다. "
        "서울의 인구는 약 950만 명으로, 대한민국 전체 인구의 약 18%를 차지한다. "
        "서울은 정치, 경제, 문화의 중심지로서 대한민국의 발전을 이끌어 왔다. "
        "또한 서울은 세계적으로도 중요한 도시로 인정받고 있으며, "
        "많은 국제기구와 다국적 기업의 본부가 위치해 있다."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="KoBART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        from transformers import PreTrainedTokenizerFast

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KoBART model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The KoBART model instance for conditional generation.
        """
        from transformers import BartForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the KoBART model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            truncation=True,
            max_length=self._variant_config.max_length,
            return_tensors="pt",
        )

        # BART conditional generation requires decoder_input_ids
        decoder_start_token_tensor = torch.tensor(
            self._cached_model.config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs

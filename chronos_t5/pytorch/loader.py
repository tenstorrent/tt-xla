# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chronos-T5 model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from chronos import ChronosPipeline

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class ChronosT5Config(ModelConfig):
    context_length: int = 512
    prediction_length: int = 64


class ModelVariant(StrEnum):
    TINY = "tiny"
    MINI = "mini"
    BASE = "base"
    LARGE = "large"
    AUTOGLUON_BASE = "autogluon-base"


class ModelLoader(ForgeModel):
    """Chronos-T5 model loader for time series forecasting.

    Loads the underlying T5ForConditionalGeneration model from the
    Chronos pipeline for single-pass encoder-decoder inference.
    """

    _VARIANTS = {
        ModelVariant.TINY: ChronosT5Config(
            pretrained_model_name="amazon/chronos-t5-tiny",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.MINI: ChronosT5Config(
            pretrained_model_name="autogluon/chronos-t5-mini",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.BASE: ChronosT5Config(
            pretrained_model_name="amazon/chronos-t5-base",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.LARGE: ChronosT5Config(
            pretrained_model_name="amazon/chronos-t5-large",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.AUTOGLUON_BASE: ChronosT5Config(
            pretrained_model_name="autogluon/chronos-t5-base",
            context_length=512,
            prediction_length=64,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chronos-T5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Chronos-T5 underlying T5 model.

        Returns:
            torch.nn.Module: The T5ForConditionalGeneration instance.
        """
        cfg = self._variant_config

        pipeline = ChronosPipeline.from_pretrained(
            cfg.pretrained_model_name,
            device_map="cpu",
            torch_dtype=dtype_override or torch.float32,
        )

        self._tokenizer = pipeline.tokenizer
        # Use the inner T5ForConditionalGeneration model directly
        # to avoid autoregressive generation in ChronosModel.forward
        model = pipeline.model.model
        model.config.use_cache = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'input_ids', 'attention_mask', and
                  'decoder_input_ids' tensors.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        # Generate a synthetic time series as sample input
        torch.manual_seed(42)
        context = torch.randn(1, cfg.context_length, dtype=dtype)

        # Tokenize the context using the pipeline's tokenizer
        token_ids, attention_mask, _ = self._tokenizer.context_input_transform(context)

        # T5 encoder-decoder requires decoder_input_ids
        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }

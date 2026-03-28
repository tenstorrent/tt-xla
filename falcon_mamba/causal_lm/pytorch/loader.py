# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-Mamba model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoTokenizer, FalconMambaForCausalLM
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Falcon-Mamba model variants for causal LM."""

    FALCON_MAMBA_TINY_DEV = "falcon_mamba_tiny_dev"


class ModelLoader(ForgeModel):
    """Falcon-Mamba model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.FALCON_MAMBA_TINY_DEV: ModelConfig(
            pretrained_model_name="tiiuae/falcon-mamba-tiny-dev",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_MAMBA_TINY_DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.text = "Hey how are you doing today?"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Falcon-Mamba",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FalconMambaForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

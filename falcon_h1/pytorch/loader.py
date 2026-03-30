# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1 model loader implementation for causal language modeling
"""
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Falcon-H1 model variants."""

    FALCON_H1_1_5B_BASE = "H1_1.5B_Base"


class ModelLoader(ForgeModel):
    """Falcon-H1 model loader implementation for causal LM tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_1_5B_BASE: ModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-1.5B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_1_5B_BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Falcon_H1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.input_text = "In a shocking discovery, scientists stumbled upon a herd of unicorns living in a remote, unexplored valley in the Andes Mountains."
        self.max_length = 512
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(
            self.input_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

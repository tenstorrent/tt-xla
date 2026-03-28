# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1 model loader implementation for causal language modeling.
"""
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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


class ModelVariant(StrEnum):
    """Available Falcon-H1 model variants."""

    FALCON_H1_0_5B_BASE = "H1_0.5B_Base"


class ModelLoader(ForgeModel):
    """Falcon-H1 model loader implementation for causal LM tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_0_5B_BASE: ModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-0.5B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_0_5B_BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="Falcon-H1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        self.text = "Hey how are you doing?"
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Falcon-H1 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Falcon-H1 model instance for causal LM.
        """
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
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Falcon-H1 model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
        )

        return inputs

    def load_config(self):
        """Load and return the configuration for the Falcon-H1 model."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-BigCode model loader implementation
"""


from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ...base import ForgeModel
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional


class ModelVariant(StrEnum):
    """Available GPT-BigCode model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-GPTBigCodeModel",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = self._variant_config.pretrained_model_name
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="GPT-BigCode",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GPT-BigCode model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The GPT-BigCode model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GPT-BigCode model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        text = "def hello_world():"
        inputs = self.tokenizer(text, return_tensors="pt")

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

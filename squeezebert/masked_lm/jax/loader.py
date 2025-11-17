# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SqueezeBERT model loader implementation for masked language modeling.
"""


from typing import Optional
import jax

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
    """Available SqueezeBERT model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """SqueezeBERT model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="squeezebert/squeezebert-uncased",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name
        self._model = None

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
            model="squeezebert",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the tokenizer for the model.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            Tokenizer: The tokenizer for the model
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}

        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the SqueezeBERT model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        from .src.model_implementation import SqueezeBertForMaskedLM
        from transformers import SqueezeBertConfig

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        config = SqueezeBertConfig.from_pretrained(self._model_name)
        self._model = SqueezeBertForMaskedLM(config)

        return self._model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SqueezeBERT model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        # Convert BatchEncoding to dict for JAX compatibility
        return dict(inputs)

    def load_parameters(self):
        """Load and return the SqueezeBERT model parameters for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            params: The loaded model parameters.
        """

        from huggingface_hub import hf_hub_download
        import torch
        import jax.numpy as jnp

        model_file = hf_hub_download(
            repo_id="squeezebert/squeezebert-uncased", filename="pytorch_model.bin"
        )
        state_dict = torch.load(model_file, weights_only=True)

        return self._model.init_from_pytorch_statedict(state_dict, dtype=jnp.bfloat16)

    def get_input_parameters(self, train=False):
        """Get input parameters for the model.

        This method provides compatibility with DynamicJaxModelTester which expects
        a get_input_parameters method.

        Args:
            train: Whether to initialize for training mode (not used for this model).

        Returns:
            PyTree: Model parameters
        """
        # Return the loaded parameters
        return self.load_parameters()

    def get_forward_method_kwargs(self, train=False):
        """Get keyword arguments for the model's forward method.

        This method ensures that the inputs are properly unpacked for the model.

        Args:
            train: Whether the model is in training mode

        Returns:
            dict: Keyword arguments for the model's forward method
        """
        # Get the inputs
        inputs = self.load_inputs()

        # Unpack the inputs and add the train flag
        kwargs = {**inputs, "train": train}

        # Add dropout RNG key for training mode
        if train:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}

        return kwargs

    def get_forward_method_args(self):
        """Get positional arguments for the model's forward method.

        For this custom Flax model using apply, we only need params as positional arg.
        The inputs will be passed as kwargs through get_forward_method_kwargs.

        Returns:
            list: List containing only the parameters
        """
        # Only return params, inputs will be in kwargs
        return [self.load_parameters()]

    def get_forward_method_name(self):
        """Get the name of the forward method for the model.

        Returns:
            str: Name of the forward method ('apply' for Flax models)
        """
        return "apply"

    def get_static_argnames(self):
        """Get static argument names for the forward method.

        Returns:
            list: List containing 'train' as a static argument
        """
        # 'train' must be static for Flax models
        return ["train"]

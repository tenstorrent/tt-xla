# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT2 model loader implementation for causal language modeling.
"""

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
from ....tools.jax_utils import cast_hf_model_to_type
import flax.nnx as nnx
from jax.sharding import PartitionSpec
import numpy as np
import jax.numpy as jnp


class ModelVariant(StrEnum):
    """Available GPT2 model variants."""

    BASE = "base"
    LARGE = "large"
    MEDIUM = "medium"
    XL = "xl"


class ModelLoader(ForgeModel):
    """GPT2 model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2-large",
        ),
        ModelVariant.MEDIUM: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2-medium",
        ),
        ModelVariant.XL: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2-xl",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "Hello there fellow traveler"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model="gpt2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the GPT2 model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from easydel import AutoEasyDeLModelForCausalLM

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        partition_rules = ((r".*", PartitionSpec()),)

        # Load the model
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name, partition_rules=partition_rules, **model_kwargs
        )

        return model

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the GPT2 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional device mesh for sharding (DataParallel mode).
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import AutoTokenizer

        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            # This matches the original test which used batch_size=8
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 8 for single device too, for consistency
            batch_size = 8

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return input_ids

    def get_input_activations_partition_spec(self, mesh, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        if np.prod(list(mesh.shape.values())) == 1:
            return PartitionSpec()

        return PartitionSpec(axis_name)

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        # Get the model state
        state = nnx.split(model_for_multichip)[1]

        partition_rules = ((r".*", PartitionSpec()),)  # Everything replicated

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules
        )

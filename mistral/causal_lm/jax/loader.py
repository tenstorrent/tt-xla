# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mistral model loader implementation for causal language modeling.
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
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available Mistral model variants."""

    V0_1 = "v0_1"
    V0_1_TINY = "v0_1_tiny"
    V0_2_INSTRUCT = "v0_2_instruct"
    V0_3_INSTRUCT = "v0_3_instruct"


class ModelLoader(ForgeModel):
    """Mistral model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V0_1: LLMModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-v0.1",
        ),
        ModelVariant.V0_1_TINY: LLMModelConfig(
            pretrained_model_name="ksmcg/Mistral-tiny",
        ),
        ModelVariant.V0_2_INSTRUCT: LLMModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        ),
        ModelVariant.V0_3_INSTRUCT: LLMModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V0_1

    sample_text = "Hello there fellow traveler"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

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
            model="mistral",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=(
                ModelSource.HUGGING_FACE
                if variant == ModelVariant.V0_1_TINY
                else ModelSource.EASYDEL
            ),
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def _is_v02_or_later(self) -> bool:
        """Check if the current variant is v0.2 or later (requires sliding window fix)."""
        return self._variant in [ModelVariant.V0_2_INSTRUCT, ModelVariant.V0_3_INSTRUCT]

    def load_model(self, dtype_override=None):
        """Load and return the Mistral model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the default dtype.
        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxMistralForCausalLM, MistralConfig
        from easydel import AutoEasyDeLModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        partition_rules = ((r".*", PartitionSpec()),)

        # For v0.2 and later models, we need to handle sliding window configuration
        if self._is_v02_or_later():
            # Initialize model with custom config to fix sliding window issue
            # From v0.2 version of Mistral-7B sliding window attention was removed,
            # but Transformers Flax implementation wasn't updated to take that into account
            config = MistralConfig.from_pretrained(pretrained_model_name)
            config.sliding_window = config.max_position_embeddings
            model = AutoEasyDeLModelForCausalLM.from_pretrained(
                pretrained_model_name,
                config=config,
                partition_rules=partition_rules,
                **model_kwargs
            )
        elif self._variant == ModelVariant.V0_1_TINY:
            # Load the model using HF for v0.1 tiny variant as there are some errors with this variant using EasyDeL
            # https://github.com/tenstorrent/tt-xla/issues/2770
            model = FlaxMistralForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        else:
            # Load the model normally for v0.1 variants
            model = AutoEasyDeLModelForCausalLM.from_pretrained(
                pretrained_model_name, partition_rules=partition_rules, **model_kwargs
            )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the Mistral model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional device mesh for sharding (DataParallel mode).
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

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

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        if self._variant == ModelVariant.V0_1_TINY:
            return inputs
        else:
            input_ids = jnp.repeat(inputs["input_ids"], batch_size, axis=0)
            return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        if np.prod(list(mesh.shape.values())) == 1:
            return (PartitionSpec(),)

        return (PartitionSpec(axis_name),)

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

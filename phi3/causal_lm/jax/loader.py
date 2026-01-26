# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI3 model loader implementation for causal language modeling using EasyDL/JAX.
"""
from typing import Optional
from transformers import AutoTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
    Parallelism,
)
from ....base import ForgeModel

import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available PHI3 model variants."""

    MINI_128K = "microsoft/Phi-3-mini-128k-instruct"
    MINI_4K = "microsoft/Phi-3-mini-4k-instruct"


class ModelLoader(ForgeModel):
    """PHI3 model loader implementation for causal LM tasks using EasyDL."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MINI_128K: ModelConfig(
            pretrained_model_name="microsoft/Phi-3-mini-128k-instruct",
        ),
        ModelVariant.MINI_4K: ModelConfig(
            pretrained_model_name="microsoft/Phi-3-mini-4k-instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MINI_128K

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="phi3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text = (
            "Can you provide ways to eat combinations of bananas and dragonfruits?"
        )
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the PHI3 model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The PHI3 model instance for causal LM.
        """
        from easydel import AutoEasyDeLModelForCausalLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the PHI3 model with default settings.

        Args:
            dtype_override: Optional dtype to override the input dtype.
            mesh: Optional device mesh for sharding.

        Returns:
            input_ids: Input tensors that can be fed to the model.
        """
        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 8 for single device too, for consistency
            batch_size = 8

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Create chat template input similar to torch version
        input_prompt = [{"role": "user", "content": self.input_text}]
        text = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=True, tokenize=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="jax",
            padding=True,
            truncation=True,
        )

        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            parallelism: The level of parallelism for sharding.
            axis_name: The name of the mesh axis to use for sharding.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        if (
            parallelism.name == Parallelism.TENSOR_PARALLEL.name
            or np.prod(list(mesh.shape.values())) == 1
        ):
            return (PartitionSpec(),)

        return (PartitionSpec(axis_name),)

    def load_parameters_partition_spec(
        self,
        model_for_multichip,
        parallelism,
        axis_name="X",
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        # Get the model state
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            # In data parallel mode, use fully replicated partitioning
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            # Use EasyDL's Phi3Config to get proper partition rules
            from easydel.modules.phi3 import Phi3Config

            phi3_config = Phi3Config()
            partition_rules = phi3_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )

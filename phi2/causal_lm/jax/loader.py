# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI2 model loader implementation for causal language modeling using EasyDL/JAX.
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
    LLMModelConfig,
    Parallelism,
)
from ....base import ForgeModel

import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available PHI2 model variants."""

    PHI2 = "microsoft/phi-2"
    PHI2_PYTDML = "microsoft/phi-2-pytdml"


class ModelLoader(ForgeModel):
    """PHI2 model loader implementation for causal LM tasks using EasyDL."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.PHI2: LLMModelConfig(
            pretrained_model_name="microsoft/phi-2",
        ),
        ModelVariant.PHI2_PYTDML: LLMModelConfig(
            pretrained_model_name="microsoft/phi-2-pytdml",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        # Determine group based on variant
        if variant == ModelVariant.PHI2:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="phi2",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text = (
            "Write a detailed analogy between mathematics and a lighthouse."
        )
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the PHI2 model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The PHI2 model instance for causal LM.
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
        """Load and return sample inputs for the PHI2 model with default settings.

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

        inputs = self.tokenizer(
            self.input_text,
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
            # Use EasyDL's PhiConfig to get proper partition rules
            from easydel.modules.phi import PhiConfig

            phi_config = PhiConfig()
            partition_rules = phi_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )

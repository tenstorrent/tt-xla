# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon model loader implementation for causal language modeling
"""
from typing import Optional
from transformers import AutoTokenizer

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
    Parallelism,
)
from ...base import ForgeModel

import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available Falcon model variants."""

    FALCON_1B = "tiiuae/Falcon3-1B-Base"
    FALCON_3B = "tiiuae/Falcon3-3B-Base"
    ## Too large
    # FALCON_7B = "tiiuae/Falcon3-7B-Base"
    # FALCON_10B = "tiiuae/Falcon3-10B-Base"


class ModelLoader(ForgeModel):
    """Falcon model loader implementation for causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FALCON_1B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-1B-Base",
        ),
        ModelVariant.FALCON_3B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-3B-Base",
        ),
        # ModelVariant.FALCON_7B: ModelConfig(
        #     pretrained_model_name="tiiuae/Falcon3-7B-Base",
        # ),
        # ModelVariant.FALCON_10B: ModelConfig(
        #     pretrained_model_name="tiiuae/Falcon3-10B-Base",
        # ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FALCON_1B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [
            ModelVariant.FALCON_1B,
            ModelVariant.FALCON_3B,
            # ModelVariant.FALCON_7B,
            # ModelVariant.FALCON_10B,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="falcon",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text_1 = "Write a function to calculate the factorial of a number"
        self.max_length = 512
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the Falcon model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Falcon model instance for causal LM.
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
        """Load and return sample inputs for the Falcon model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
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

        tokenizer_kwargs = {}

        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        input_ids = self.tokenizer.encode(
            self.input_text_1,
            add_special_tokens=True,
            return_tensors="jax",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = jnp.repeat(input_ids, batch_size, axis=0)

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
            # Use EasyDel's FalconConfig to get proper partition rules
            from easydel.modules.falcon import FalconConfig

            falcon_config = FalconConfig()
            partition_rules = falcon_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )

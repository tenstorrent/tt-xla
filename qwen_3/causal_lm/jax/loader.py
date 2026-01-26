# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 model loader implementation for causal language modeling using EasyDL/JAX.
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
    """Available Qwen3 model variants for causal language modeling."""

    QWEN_3_0_6B = "0_6b"
    QWEN_3_1_7B = "1_7b"
    QWEN_3_4B = "4b"
    ## Too large
    # QWEN_3_8B = "8b"
    # QWEN_3_14B = "14b"
    # QWEN_3_32B = "32b"
    # QWEN_3_30B_A3B = "30b_a3b"


class ModelLoader(ForgeModel):
    """Qwen3 model loader implementation for causal LM tasks using EasyDL."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_0_6B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-0.6B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_1_7B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-1.7B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_4B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-4B",
            max_length=128,
        ),
        # ModelVariant.QWEN_3_8B: LLMModelConfig(
        #     pretrained_model_name="Qwen/Qwen3-8B",
        #     max_length=128,
        # ),
        # ModelVariant.QWEN_3_14B: LLMModelConfig(
        #     pretrained_model_name="Qwen/Qwen3-14B",
        #     max_length=128,
        # ),
        # ModelVariant.QWEN_3_32B: LLMModelConfig(
        #     pretrained_model_name="Qwen/Qwen3-32B",
        #     max_length=128,
        # ),
        # ModelVariant.QWEN_3_30B_A3B: LLMModelConfig(
        #     pretrained_model_name="Qwen/Qwen3-30B-A3B",
        #     max_length=128,
        # ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_0_6B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        # Determine group based on variant size
        if variant in [
            ModelVariant.QWEN_3_0_6B,
            ModelVariant.QWEN_3_1_7B,
            ModelVariant.QWEN_3_4B,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="qwen_3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text = "Give me a short introduction to large language model."
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the Qwen3 model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The Qwen3 model instance for causal LM.
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
        """Load and return sample inputs for the Qwen3 model with default settings.

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

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        input_ids = self.tokenizer.encode(
            self.input_text,
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
            # Use EasyDL's Qwen3Config to get proper partition rules
            from easydel.modules.qwen3 import Qwen3Config

            qwen3_config = Qwen3Config()
            partition_rules = qwen3_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2.5-Coder model loader implementation for causal language modeling using EasyDL/JAX.
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
    """Available Qwen2.5-Coder model variants for causal language modeling."""

    QWEN_2_5_CODER_0_5B = "0_5b"
    QWEN_2_5_CODER_1_5B = "1_5b"
    QWEN_2_5_CODER_1_5B_INSTRUCT = "1_5b_instruct"
    QWEN_2_5_CODER_3B = "3b"
    QWEN_2_5_CODER_3B_INSTRUCT = "3b_instruct"
    QWEN_2_5_CODER_7B = "7b"
    QWEN_2_5_CODER_7B_INSTRUCT = "7b_instruct"
    ## Too large
    # QWEN_2_5_CODER_32B_INSTRUCT = "32b_instruct"


class ModelLoader(ForgeModel):
    """Qwen2.5-Coder model loader implementation for causal LM tasks using EasyDL."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_5_CODER_0_5B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-0.5B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_1_5B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-1.5B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-3B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-7B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_length=128,
        ),
        # ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT: LLMModelConfig(
        #     pretrained_model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        #     max_length=128,
        # ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_CODER_0_5B

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
            ModelVariant.QWEN_2_5_CODER_0_5B,
            ModelVariant.QWEN_2_5_CODER_1_5B,
            ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT,
            ModelVariant.QWEN_2_5_CODER_3B,
            ModelVariant.QWEN_2_5_CODER_3B_INSTRUCT,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="qwen_2_5_coder",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters - coding focused prompt
        self.input_text = "def fibonacci(n):"
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load and return the Qwen2.5-Coder model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The Qwen2.5-Coder model instance for causal LM.
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
        """Load and return sample inputs for the Qwen2.5-Coder model with default settings.

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
            # Use EasyDL's Qwen2Config to get proper partition rules (Coder uses same architecture as Qwen2)
            from easydel.modules.qwen2 import Qwen2Config

            qwen2_config = Qwen2Config()
            partition_rules = qwen2_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )

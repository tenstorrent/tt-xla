# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import MixtralConfig

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model import FlaxMixtralForCausalLM


class ModelVariant(StrEnum):
    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="mistralai/Mixtral-8x7B-v0.1",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="mistralai/Mixtral-8x7B-v0.1",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="mistralai/Mixtral-8x7B-v0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CUSTOM_1X2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mixtral-8x7B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    @staticmethod
    def _set_config() -> MixtralConfig:
        config = MixtralConfig()
        config.mesh = None

        # config must have set_model_mesh from jax_workload
        def set_model_mesh(mesh):
            config.mesh = mesh

        config.set_model_mesh = set_model_mesh
        config.num_hidden_layers = 2
        config.intermediate_size = 1024
        config.head_dim = 128  # Default is None
        return config

    def _get_config(self) -> MixtralConfig:
        if self.config is None:
            self.config = self._set_config()
        return self.config

    def load_model(self, *, dtype_override=None):
        config = self._get_config()
        return FlaxMixtralForCausalLM(
            config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)
        )

    def load_inputs(self, dtype_override=None, mesh=None):
        config = self._get_config()
        rng = np.random.default_rng(42)
        input_ids = jnp.array(rng.integers(1, 1000, size=(8, 8), dtype=np.int32))
        attention_mask = jnp.ones((8, 8), dtype=jnp.int32)
        past_key_values = {
            f"layer_{i}": {
                "cached_key": jnp.zeros(
                    (8, 8, config.num_key_value_heads, config.head_dim),
                    dtype=jnp.float32,
                ),
                "cached_value": jnp.zeros(
                    (8, 8, config.num_key_value_heads, config.head_dim),
                    dtype=jnp.float32,
                ),
                "cache_index": jnp.array(0, dtype=jnp.int32),
            }
            for i in range(config.num_hidden_layers)
        }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def load_parameters(
        self,
        dtype_override=None,
        train=False,
        seed=None,
        inputs=None,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        input_parameters_partition_specs=None,
        **_,
    ):
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model()
        )
        return nnx.split(model)[1]

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        parallelism=None,
        dtype_override=None,
        **_,
    ):
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model()
        )
        _, state = nnx.split(model)
        return nnx.get_partition_spec(state)

    def get_input_activations_partition_spec(
        self, mesh, axis_name="X", parallelism=None, **_
    ):
        inputs = self.load_inputs()
        return tuple(PartitionSpec() for _ in inputs)

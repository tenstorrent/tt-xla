# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from transformers import LlamaConfig  # Original used config from LlamaConfig
from .src.model import FlaxFalcon3ForCausalLMModule


class _FalconWrapper(linen.Module):
    config: Any
    dtype: Any

    def setup(self):
        self.llm = FlaxFalcon3ForCausalLMModule(config=self.config, dtype=self.dtype)

    def __call__(self, inputs):
        input_ids, attention_mask, position_ids = inputs
        return self.llm(input_ids, attention_mask, position_ids)


class ModelVariant(StrEnum):

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
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
            model="Falcon3-7B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    @staticmethod
    def _set_config() -> LlamaConfig:
        config = LlamaConfig()
        config.mesh = None

        # config must have set_model_mesh from jax_workload
        def set_model_mesh(mesh):
            config.mesh = mesh

        config.set_model_mesh = set_model_mesh
        config.num_hidden_layers = 2
        return config

    def _get_config(self) -> LlamaConfig:
        if self.config is None:
            self.config = self._set_config()
        return self.config

    def load_model(self, *, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else jnp.bfloat16
        return _FalconWrapper(config=self._get_config(), dtype=dtype)

    def load_inputs(self, dtype_override=None, mesh=None):
        rng = np.random.default_rng(42)
        input_ids = jnp.array(rng.integers(1, 1000, size=(8, 8), dtype=np.int32))
        attention_mask = jnp.ones((8, 8), dtype=jnp.int32)
        position_ids = jnp.broadcast_to(jnp.arange(8)[None, :], (8, 8))
        return (input_ids, attention_mask, position_ids)

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
    ):
        from infra.utilities import initialize_flax_linen_parameters_on_cpu

        if inputs is None:
            inputs = self.load_inputs(mesh=cpu_mesh)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        return initialize_flax_linen_parameters_on_cpu(
            model,
            input_activations_partition_specs,
            inputs,
            input_parameters_partition_specs,
            cpu_mesh,
            seed if seed is not None else 42,
        )

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        parallelism=None,
        dtype_override=None,
    ):
        from infra.utilities import make_flax_linen_parameters_partition_specs_on_cpu

        if inputs is None:
            inputs = self.load_inputs(mesh=cpu_mesh)
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model(dtype_override=dtype_override)
        )
        return make_flax_linen_parameters_partition_specs_on_cpu(
            model, cpu_mesh, input_activations_partition_specs, inputs
        )

    def get_input_activations_partition_spec(
        self, mesh, axis_name="X", parallelism=None
    ):
        from jax.sharding import PartitionSpec

        return (PartitionSpec(),)

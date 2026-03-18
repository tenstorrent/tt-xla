# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec

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
from .src.model import Gemma3ForCausalLM
from transformers import Gemma3TextConfig, MistralConfig


class ModelVariant(StrEnum):

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="google/gemma-3-4b",
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
            model="Gemma3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    @staticmethod
    def _set_config() -> Gemma3TextConfig:
        config = Gemma3TextConfig()
        config.mesh = None

        # config must have set_model_mesh from jax_workload
        def set_model_mesh(mesh):
            config.mesh = mesh

        config.set_model_mesh = set_model_mesh
        config.num_hidden_layers = 2
        config.intermediate_size = 1024
        config.hidden_size = 512

        config.vocab_size = 4096  # without this we get OOM error

        # model implementation specific
        config.dtype = jnp.float32
        config.mesh = None
        config.param_dtype = jnp.bfloat16
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        config.rope_local_base_freq = 10_000.0
        config.query_pre_attn_scalar = 256.0
        config.final_logit_soft_cap = None
        config.attn_logit_soft_cap = None
        config.hidden_activation = "gelu_pytorch_tanh"
        config.use_cache = False

        return config

    def _get_config(self) -> Gemma3TextConfig:
        if self.config is None:
            self.config = self._set_config()
        return self.config

    def load_model(self, *, dtype_override=None):
        config = self._get_config()
        if dtype_override is not None:
            config.dtype = dtype_override
            config.param_dtype = dtype_override
        return Gemma3ForCausalLM(config, rngs=nnx.Rngs(0))

    def load_inputs(self, dtype_override=None, mesh=None):
        rng = np.random.default_rng(42)
        input_ids = jnp.array(rng.integers(1, 1000, size=(8, 8), dtype=np.int32))
        return {"input_ids": input_ids}

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
    ):
        model = (
            model_for_multichip
            if model_for_multichip is not None
            else self.load_model()
        )
        _, state = nnx.split(model)
        return nnx.get_partition_spec(state)

    def get_input_activations_partition_spec(
        self, mesh, axis_name="X", parallelism=None
    ):
        inputs = self.load_inputs()
        return tuple(PartitionSpec() for _ in inputs)

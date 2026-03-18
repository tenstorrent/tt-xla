# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import MistralConfig

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
from .src.model import Axis, MistralModel

_TP_SHARDING_RULES = [
    (Axis.QHEAD, None),
    (Axis.KVHEAD, None),
    (Axis.MLP, "X"),
    (Axis.EMBED, None),
    (Axis.VOCAB, None),
    (Axis.HEAD, "X"),
]


class ModelVariant(StrEnum):

    CUSTOM_1X2 = "Custom_1x2"
    CUSTOM_1X4 = "Custom_1x4"
    CUSTOM_1X8 = "Custom_1x8"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-3.1-24B-Base-2503",
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
            model="Mistral-Small-3.1-24B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    @staticmethod
    def _set_config() -> MistralConfig:
        config = MistralConfig()
        config.mesh = None

        # config must have set_model_mesh from jax_workload
        def set_model_mesh(mesh):
            config.mesh = mesh

        config.set_model_mesh = set_model_mesh
        config.num_hidden_layers = 10
        config.head_dim = 128  # Default is None
        return config

    def _get_config(self) -> MistralConfig:
        if self.config is None:
            self.config = self._set_config()
        return self.config

    def load_model(self, *, dtype_override=None):
        config = self._get_config()
        return MistralModel(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(0),
            sharding_rules=_TP_SHARDING_RULES,
        )

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
        if model_for_multichip is not None:
            model = model_for_multichip
        else:
            model = self.load_model(dtype_override=dtype_override)
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
        if model_for_multichip is not None:
            model = model_for_multichip
        else:
            model = self.load_model(dtype_override=dtype_override)
        _, state = nnx.split(model)
        rules_dict = {str(axis): mesh_axis for axis, mesh_axis in _TP_SHARDING_RULES}

        def make_spec(variable):
            logical = getattr(variable, "sharding", None)
            if logical is None:
                return PartitionSpec()
            physical = tuple(rules_dict.get(str(ax), None) for ax in logical)
            return PartitionSpec(*physical)

        flat = state.flat_state()
        specs = [make_spec(var) for _, var in flat]
        flat_specs = nnx.graph.FlatState.from_sorted_keys_values(flat.paths, specs)

        return flat_specs.to_nested_state()

    def get_input_activations_partition_spec(
        self, mesh, axis_name="X", parallelism=None
    ):
        inputs = self.load_inputs()
        return tuple(PartitionSpec() for _ in inputs)

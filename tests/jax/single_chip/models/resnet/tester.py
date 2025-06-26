# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    FlaxPreTrainedModel,
    FlaxResNetForImageClassification,
)
from utils import StrEnum

from tests.jax.single_chip.models.model_utils import torch_statedict_to_pytree


class ResNetTester(ModelTester):
    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        # Resnet-50 has a flax checkpoint on HF, so we can just load it directly.
        hf_path = f"microsoft/{self._model_variant}"
        return FlaxResNetForImageClassification.from_pretrained(hf_path)

    # @override
    def _get_input_activations(self) -> jax.Array:
        return jax.random.uniform(jax.random.PRNGKey(0), (1, 3, 224, 224))

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {
            "params": self._input_parameters,
            "pixel_values": self._input_activations,
        }

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]

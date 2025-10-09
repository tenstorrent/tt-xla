# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, Model, RunMode
from infra.workloads.workload import Workload
from jaxtyping import PyTree

from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelArchitecture,
    ModelLoader,
)


class MNISTCNNTester(JaxModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        variant: ModelArchitecture,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant)
        self._is_dropout = variant == ModelArchitecture.CNN_DROPOUT
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model_loader.load_parameters()

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        kwargs = {"train": (False if self._run_mode == RunMode.INFERENCE else True)}
        if self._is_dropout and self._run_mode == RunMode.TRAINING:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}
        return kwargs

    # @override
    def _get_static_argnames(self):
        return ["train"]

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()
        forward_static_args = self._get_static_argnames()
        forward_method_name = self._get_forward_method_name()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"
        assert hasattr(
            self._model, forward_method_name
        ), f"Model does not have {forward_method_name} method provided."

        forward_pass_method = getattr(self._model, forward_method_name)

        if self._run_mode == RunMode.TRAINING:
            forward_pass_method = lambda params, inputs, **kwargs: self._model.apply(
                params, inputs, mutable=["batch_stats"], **kwargs
            )[0]
        else:
            forward_pass_method = getattr(self._model, forward_method_name)

        self._workload = Workload(
            framework=self._framework,
            executable=forward_pass_method,
            args=args,
            kwargs=kwargs,
            static_argnames=forward_static_args,
        )

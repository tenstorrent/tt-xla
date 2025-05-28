# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, MultichipModelTester, RunMode
from infra.device_connector import device_connector
from infra.multichip_utils import (
    initialize_flax_linen_parameters_on_cpu,
    make_flax_linen_parameters_partition_specs_on_cpu,
)
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from tests.jax.single_chip.models.mnist.mlp.tester import (
    MNIST_MLP_PARAMS_INIT_SEED,
    create_mnist_random_input_image,
)

from .model_implementation import MNISTMLPMultichipModel


class MnistMLPMultichipTester(MultichipModelTester):
    """Tester for multichip versions of MNIST MLP model."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        run_mode: RunMode,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        num_devices: int = device_connector.get_number_of_tt_devices(),
    ) -> None:
        self._hidden_sizes = hidden_sizes
        self.main_axis_name = "X"
        self.num_devices = num_devices

        mesh_shape = (self.num_devices,)
        axis_names = (self.main_axis_name,)

        super().__init__(mesh_shape, axis_names, comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return MNISTMLPMultichipModel(
            hidden_sizes=self._hidden_sizes,
            axis_name=self.main_axis_name,
            num_devices=self.num_devices,
            train_mode=self._run_mode == RunMode.TRAINING,
        )

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations_partition_specs(self) -> PartitionSpec:
        # Sharding data on batch axis since data parallelism is utilized for the
        # convolutional layers.
        return PartitionSpec(self.main_axis_name)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return create_mnist_random_input_image()

    # @override
    def _get_input_parameters_partition_specs(self) -> PyTree:
        return make_flax_linen_parameters_partition_specs_on_cpu(
            self._model,
            self.cpu_mesh,
            self._input_activations_partition_specs,
            self._input_activations,
        )

    # @override
    def _get_input_parameters(self) -> PyTree:
        return initialize_flax_linen_parameters_on_cpu(
            self._model,
            self._input_activations_partition_specs,
            self._input_activations,
            self._input_parameters_partition_specs,
            self.cpu_mesh,
            MNIST_MLP_PARAMS_INIT_SEED,
        )

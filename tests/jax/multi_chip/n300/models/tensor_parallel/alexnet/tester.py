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
    make_flax_linen_parameters_partition_specs,
)
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from .model_implementation import AlexNetMultichipModel


class AlexNetMultichipTester(MultichipModelTester):
    """Tester for multichip versions of AlexNet CNN model."""

    def __init__(
        self,
        run_mode: RunMode,
        comparison_config: ComparisonConfig = ComparisonConfig(),
    ) -> None:
        self.main_axis_name = "X"
        self.num_devices = device_connector.get_number_of_tt_devices()

        # Currently we support only 2D mesh with shardy enabled.
        mesh_shape = (1, self.num_devices)
        axis_names = ("Y", self.main_axis_name)

        super().__init__(mesh_shape, axis_names, comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return AlexNetMultichipModel(
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
        return jax.random.randint(
            key=jax.random.PRNGKey(23),
            # B, H, W, C
            shape=(4, 224, 224, 3),
            # In the original paper inputs are normalized with individual channel
            # values learned from training set.
            minval=-128,
            maxval=128,
        )

    # @override
    def _get_input_parameters_partition_specs(self) -> PyTree:
        return make_flax_linen_parameters_partition_specs(
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
        )

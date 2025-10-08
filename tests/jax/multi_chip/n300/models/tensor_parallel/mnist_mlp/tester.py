# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence

import jax
from flax import linen as nn
from infra import (
    ComparisonConfig,
    DeviceConnectorFactory,
    Framework,
    JaxMultichipModelTester,
    RunMode,
    initialize_flax_linen_parameters_on_cpu,
    make_flax_linen_parameters_partition_specs_on_cpu,
)
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from tests.jax.single_chip.models.mnist.mlp.tester import (
    MNIST_MLP_PARAMS_INIT_SEED,
    create_mnist_random_input_image,
)

from third_party.tt_forge_models.mnist.image_classification.jax import (
    MNISTMLPMultichipModel,
)


class MnistMLPMultichipTester(JaxMultichipModelTester):
    """Tester for multichip versions of MNIST MLP model."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        run_mode: RunMode,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        num_devices: Optional[int] = None,
    ) -> None:
        if num_devices is not None:
            self.num_devices = num_devices
        else:
            device_connector = DeviceConnectorFactory.create_connector(Framework.JAX)
            self.num_devices = device_connector.get_number_of_tt_devices()

        self._hidden_sizes = hidden_sizes
        self.main_axis_name = "X"
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
    def _get_input_activations_partition_spec(self) -> PartitionSpec:
        # No data parallelism utilized in this model.
        return PartitionSpec()

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return create_mnist_random_input_image()

    # @override
    def _get_input_parameters_partition_spec(self) -> PyTree:
        return make_flax_linen_parameters_partition_specs_on_cpu(
            self._model,
            self._cpu_mesh,
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
            self._cpu_mesh,
            MNIST_MLP_PARAMS_INIT_SEED,
        )

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Torch model tester implementation."""

import collections
from typing import Any

import torch
import torch_xla.runtime as xr
from infra.comparators import ComparisonConfig
from infra.testers.single_chip.model import RunMode, TorchModelTester
from infra.utilities.torch_multichip_utils import get_mesh

from tests.runner.utils import TorchDynamicLoader
from third_party.tt_forge_models.config import Parallelism


class DynamicTorchModelTester(TorchModelTester):
    """Torch model tester that uses a dynamic loader for model and input loading.

    This tester delegates model and input loading to a loader object, allowing
    for flexible model loading without subclassing for each model variant.
    """

    def __init__(
        self,
        run_mode: RunMode,
        *,
        loader,
        comparison_config: ComparisonConfig | None = None,
        parallelism: Parallelism = Parallelism.SINGLE_DEVICE,
    ) -> None:
        """Initialize DynamicTorchModelTester.

        Args:
            run_mode: RunMode.INFERENCE or RunMode.TRAINING
            loader: Loader object that implements load_model and load_inputs methods
            comparison_config: Optional comparison configuration for result validation
            parallelism: Parallelism mode for model execution
        """
        # Create TorchDynamicLoader instance
        self.dynamic_loader = TorchDynamicLoader(loader)
        # Store parallelism for reporting/consumers
        self.parallelism = parallelism

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            run_mode=run_mode,
            parallelism=self.parallelism,
        )

    # --- TorchModelTester interface implementations ---

    def _get_model(self):
        """Get model instance from the dynamic loader.

        Returns:
            Model instance loaded from the loader
        """
        return self.dynamic_loader.load_model()

    def _get_input_activations(self):
        """Get input activations from the dynamic loader.

        Returns:
            Input tensors loaded from the loader
        """
        inputs = self.dynamic_loader.load_inputs()

        if self.parallelism == Parallelism.DATA_PARALLEL:
            num_devices = xr.global_runtime_device_count()
            if isinstance(inputs, collections.abc.Mapping):
                inputs = {
                    k: self.dynamic_loader.batch_tensor(v, num_devices)
                    for k, v in inputs.items()
                }
            elif isinstance(inputs, collections.abc.Sequence):
                inputs = [
                    self.dynamic_loader.batch_tensor(inp, num_devices) for inp in inputs
                ]
            else:
                inputs = self.dynamic_loader.batch_tensor(inputs, num_devices)

        return inputs

    def _get_shard_specs_function(self):
        """Get shard specs function from the dynamic loader if available.

        Returns:
            Shard spec function if loader supports it, None otherwise
        """
        if self.parallelism == Parallelism.DATA_PARALLEL:
            return self.dynamic_loader.load_shard_spec_data_parallel
        else:
            return self.dynamic_loader.get_shard_spec_function()

    def _get_mesh(self):
        """Get mesh configuration from the dynamic loader if available.

        Returns:
            Mesh object if loader supports mesh configuration, None otherwise
        """
        num_devices = xr.global_runtime_device_count()
        if self.parallelism == Parallelism.DATA_PARALLEL:
            mesh_shape, mesh_names = (1, num_devices), ("model", "data")
        else:
            mesh_shape, mesh_names = self.dynamic_loader.get_mesh_config(num_devices)

        if mesh_shape and mesh_names:
            return get_mesh(mesh_shape, mesh_names)
        return None

    def _unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unwraps model output to a single tensor.
        Calls the unpack_forward_output method of the dynamic loader.
        """
        return self.dynamic_loader.unpack_forward_output(output)

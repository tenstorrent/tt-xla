# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Torch model tester implementation."""

import collections
from typing import Any

import torch
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig
from infra.testers.compiler_config import CompilerConfig
from infra.testers.single_chip.model import RunMode, TorchModelTester
from infra.utilities.torch_multichip_utils import get_mesh

from tests.runner.test_utils import RunPhase
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
        compiler_config: CompilerConfig = None,
        parallelism: Parallelism = Parallelism.SINGLE_DEVICE,
        run_phase: RunPhase = RunPhase.DEFAULT,
        test_metadata=None,
    ) -> None:
        """Initialize DynamicTorchModelTester.

        Args:
            run_mode: RunMode.INFERENCE or RunMode.TRAINING
            loader: Loader object that implements load_model and load_inputs methods
            comparison_config: Optional comparison configuration for result validation
            parallelism: Parallelism mode for model execution
            run_phase: Optional run phase (DEFAULT, LLM_DECODE, LLM_PREFILL)
            test_metadata: Optional ModelTestConfig with seq_len/batch_size for prefill
        """
        # Create TorchDynamicLoader instance
        self.dynamic_loader = TorchDynamicLoader(loader)
        # Store parallelism for reporting/consumers
        self.parallelism = parallelism
        # Store phase hint for input loading
        self.run_phase = run_phase
        # Store test metadata for seq_len/batch_size access
        self._test_metadata = test_metadata

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            compiler_config=compiler_config,
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
        # Extract seq_len and batch_size from test_metadata if available
        seq_len = (
            getattr(self._test_metadata, "seq_len", None)
            if self._test_metadata
            else None
        )
        batch_size = (
            getattr(self._test_metadata, "batch_size", None)
            if self._test_metadata
            else None
        )

        inputs = self.dynamic_loader.load_inputs(
            run_phase=self.run_phase,
            seq_len=seq_len,
            batch_size=batch_size,
        )

        # Determine input batching factor:
        #   DATA_PARALLEL  → replicate across all devices
        #   shard_inputs   → replicate across the batch axis of the mesh (e.g. 2 for a (2,4) mesh)
        batch_factor = None
        if self.parallelism == Parallelism.DATA_PARALLEL:
            batch_factor = xr.global_runtime_device_count()
        else:
            shard_inputs = getattr(self._test_metadata, "shard_inputs", None) if self._test_metadata else None
            mesh_shape = getattr(self._test_metadata, "mesh_shape", None) if self._test_metadata else None
            if shard_inputs and mesh_shape is not None and mesh_shape[0] > 1:
                batch_factor = mesh_shape[0]

        if batch_factor is not None:
            if isinstance(inputs, collections.abc.Mapping):
                inputs = {
                    k: self.dynamic_loader.batch_tensor(v, batch_factor)
                    for k, v in inputs.items()
                }
            elif isinstance(inputs, collections.abc.Sequence):
                inputs = [
                    self.dynamic_loader.batch_tensor(inp, batch_factor) for inp in inputs
                ]
            else:
                inputs = self.dynamic_loader.batch_tensor(inputs, batch_factor)

        return inputs

    def _get_shard_specs_function(self):
        """Get shard specs function from the dynamic loader if available.

        Returns:
            Shard spec function if loader supports it, None otherwise
        """
        if self.parallelism == Parallelism.DATA_PARALLEL:
            return self.dynamic_loader.load_shard_spec_data_parallel

        # Check for explicit shard_strategy override from test_metadata
        shard_strategy = getattr(self._test_metadata, "shard_strategy", None) if self._test_metadata else None
        shard_inputs = getattr(self._test_metadata, "shard_inputs", None) if self._test_metadata else None

        if shard_strategy is not None:
            # Build a custom shard spec function that uses the requested strategy.
            # The function signature determines how the runner calls it:
            #   (model)              → weight sharding only
            #   (model, args, kwargs) → weight + input (activation) sharding
            loader = self.dynamic_loader.loader
            dynamic_loader = self.dynamic_loader

            if shard_inputs:
                def shard_spec_fn(model, args, kwargs):
                    specs = loader.load_shard_spec(model, strategy=shard_strategy) or {}
                    # Reuse existing input-sharding logic (shards dim 0 on "data" axis)
                    input_specs = dynamic_loader.load_shard_spec_data_parallel(
                        args, kwargs
                    )
                    specs.update(input_specs)
                    return specs
            else:
                def shard_spec_fn(model):
                    return loader.load_shard_spec(model, strategy=shard_strategy)

            return shard_spec_fn

        # Default: existing loader-driven behavior
        return self.dynamic_loader.get_shard_spec_function()

    def _get_mesh(self):
        """Get mesh configuration from the dynamic loader if available.

        Returns:
            Mesh object if loader supports mesh configuration, None otherwise
        """
        if self.parallelism == Parallelism.SINGLE_DEVICE:
            return None

        num_devices = xr.global_runtime_device_count()
        if self.parallelism == Parallelism.DATA_PARALLEL:
            mesh_shape, mesh_names = (1, num_devices), ("model", "data")
        else:
            # Check for mesh_shape override from test_metadata (set by ShardingConfig)
            mesh_shape_override = (
                getattr(self._test_metadata, "mesh_shape", None)
                if self._test_metadata
                else None
            )
            if mesh_shape_override is not None:
                mesh_shape, mesh_names = mesh_shape_override, ("data", "model")
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

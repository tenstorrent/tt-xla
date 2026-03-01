# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Torch model tester implementation."""

import collections
import inspect
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

        Handles standard TP, DP, and combined TP+DP with FSDP/megatron strategies.

        Returns:
            Shard spec function if loader supports it, None otherwise
        """
        if self.parallelism == Parallelism.DATA_PARALLEL:
            return self.dynamic_loader.load_shard_spec_data_parallel

        strategy = (
            getattr(self._test_metadata, "sharding_strategy", None)
            if self._test_metadata
            else None
        )
        shard_inputs = (
            getattr(self._test_metadata, "shard_inputs", False)
            if self._test_metadata
            else False
        )

        # No explicit strategy → default TP behavior from the loader
        if strategy is None:
            return self.dynamic_loader.get_shard_spec_function()

        loader_shard_spec_fn = self.dynamic_loader.get_shard_spec_function()
        if loader_shard_spec_fn is None:
            return None

        supports_strategy_kwargs = False
        try:
            signature = inspect.signature(self.dynamic_loader.loader.load_shard_spec)
            params = signature.parameters
            supports_strategy_kwargs = "strategy" in params and "batch_axis" in params
        except (TypeError, ValueError):
            supports_strategy_kwargs = False

        # Build the weight shard spec function.
        # When shard_inputs is on, the mesh uses ("data", "model") axes, so FSDP
        # specs must use "data" instead of "batch" — handled by batch_axis arg.
        # (Megatron ignores batch_axis — its non-model axis is always None.)
        batch_axis = "data" if shard_inputs else "batch"
        if supports_strategy_kwargs:
            weight_fn = lambda model: self.dynamic_loader.loader.load_shard_spec(
                model, strategy=strategy, batch_axis=batch_axis
            )
        else:
            # Backward-compat fallback: ignore explicit strategy if loader does
            # not support strategy kwargs.
            weight_fn = loader_shard_spec_fn

        if shard_inputs:
            # Combined TP + input sharding: shard weights AND inputs.
            # The device runner dispatches to the 3-arg form (model, args, kwargs).
            dp_loader = self.dynamic_loader

            def combined_shard_spec(model, args, kwargs):
                weight_specs = weight_fn(model) or {}
                dp_specs = dp_loader.load_shard_spec_data_parallel(args, kwargs)
                weight_specs.update(dp_specs)
                return weight_specs

            return combined_shard_spec

        return weight_fn

    def _get_mesh(self):
        """Get mesh configuration from the dynamic loader if available.

        Supports explicit mesh_shape / shard_inputs from test_metadata.

        Returns:
            Mesh object if loader supports mesh configuration, None otherwise
        """
        if self.parallelism == Parallelism.SINGLE_DEVICE:
            return None

        num_devices = xr.global_runtime_device_count()

        if self.parallelism == Parallelism.DATA_PARALLEL:
            mesh_shape, mesh_names = (1, num_devices), ("model", "data")
        elif self._test_metadata and getattr(self._test_metadata, "mesh_shape", None):
            # Explicit mesh from test metadata, e.g. (1, 8) or (2, 4)
            mesh_shape = self._test_metadata.mesh_shape
            shard_inputs = getattr(self._test_metadata, "shard_inputs", False)
            if shard_inputs:
                mesh_names = ("data", "model")
            else:
                mesh_names = ("batch", "model")
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

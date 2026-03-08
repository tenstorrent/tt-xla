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
from loguru import logger
from tt_torch.sparse_mlp import enable_sparse_mlp, get_moe_shard_specs

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
        print(f"\n{'='*80}", flush=True)
        print(f"[DEBUG][DynamicTorchModelTester.__init__] CALLED", flush=True)
        print(f"  run_mode = {run_mode}", flush=True)
        print(f"  loader = {type(loader).__name__}", flush=True)
        print(f"  comparison_config = {comparison_config}", flush=True)
        print(f"  compiler_config = {compiler_config}", flush=True)
        print(f"  parallelism = {parallelism}", flush=True)
        print(f"  run_phase = {run_phase}", flush=True)
        print(f"  test_metadata = {test_metadata}", flush=True)
        print(f"{'='*80}", flush=True)

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
        print(f"[DEBUG][DynamicTorchModelTester.__init__] DONE", flush=True)

        if test_metadata and getattr(test_metadata, "inject_custom_moe", False):
            self._inject_custom_moe(self._model)

    # --- TorchModelTester interface implementations ---

    def _get_model(self):
        """Get model instance from the dynamic loader.

        Returns:
            Model instance loaded from the loader
        """
        print(f"\n[DEBUG][DynamicTorchModelTester._get_model] CALLED — loading model via dynamic_loader", flush=True)
        model = self.dynamic_loader.load_model()
        print(f"[DEBUG][DynamicTorchModelTester._get_model] Loaded model: type={type(model).__name__}", flush=True)
        return model

    def _get_input_activations(self):
        """Get input activations from the dynamic loader.

        Returns:
            Input tensors loaded from the loader
        """
        print(f"\n[DEBUG][DynamicTorchModelTester._get_input_activations] CALLED", flush=True)
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
        print(f"[DEBUG][DynamicTorchModelTester._get_input_activations] run_phase={self.run_phase}, seq_len={seq_len}, batch_size={batch_size}", flush=True)

        inputs = self.dynamic_loader.load_inputs(
            run_phase=self.run_phase,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        print(f"[DEBUG][DynamicTorchModelTester._get_input_activations] Loaded inputs: type={type(inputs).__name__}", flush=True)

        if self.parallelism == Parallelism.DATA_PARALLEL:
            num_devices = xr.global_runtime_device_count()
            print(f"[DEBUG][DynamicTorchModelTester._get_input_activations] DATA_PARALLEL: batching for {num_devices} devices", flush=True)
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
        print(f"[DEBUG][DynamicTorchModelTester._get_shard_specs_function] CALLED — parallelism={self.parallelism}", flush=True)
        if self.parallelism == Parallelism.DATA_PARALLEL:
            print(f"[DEBUG][DynamicTorchModelTester._get_shard_specs_function] Returning data_parallel shard spec", flush=True)
            return self.dynamic_loader.load_shard_spec_data_parallel
        else:
            result = self.dynamic_loader.get_shard_spec_function()
            print(f"[DEBUG][DynamicTorchModelTester._get_shard_specs_function] Returning: {'set' if result else None}", flush=True)
            return result

    def _get_mesh(self):
        """Get mesh configuration from the dynamic loader if available.

        Returns:
            Mesh object if loader supports mesh configuration, None otherwise
        """
        print(f"[DEBUG][DynamicTorchModelTester._get_mesh] CALLED — parallelism={self.parallelism}", flush=True)
        if self.parallelism == Parallelism.SINGLE_DEVICE:
            print(f"[DEBUG][DynamicTorchModelTester._get_mesh] SINGLE_DEVICE — returning None", flush=True)
            return None

        num_devices = xr.global_runtime_device_count()
        if self.parallelism == Parallelism.DATA_PARALLEL:
            mesh_shape, mesh_names = (1, num_devices), ("model", "data")
        else:
            mesh_shape, mesh_names = self.dynamic_loader.get_mesh_config(num_devices)

        print(f"[DEBUG][DynamicTorchModelTester._get_mesh] mesh_shape={mesh_shape}, mesh_names={mesh_names}", flush=True)
        if mesh_shape and mesh_names:
            return get_mesh(mesh_shape, mesh_names)
        return None

    def _unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unwraps model output to a single tensor.
        Calls the unpack_forward_output method of the dynamic loader.
        """
        print(f"[DEBUG][DynamicTorchModelTester._unpack_forward_output] CALLED — output type={type(output).__name__}", flush=True)
        return self.dynamic_loader.unpack_forward_output(output)

    def _inject_custom_moe(self, model):
        """Injects a custom MoE implementation into the model if specified in test metadata."""
        logger.info(
            "Custom MoE injection enabled for this test - using sparse_mlp.py implementation in tt_torch"
        )
        mesh_info = self._workload.mesh.shape()
        mesh_shape = tuple(mesh_info.values())
        mesh_names = tuple(mesh_info.keys())
        enable_sparse_mlp(model, mesh=mesh_shape)
        shard_spec_fn = self._workload.shard_spec_fn
        if shard_spec_fn:

            def combined_shard_spec_fn(model, _fn=shard_spec_fn, _names=mesh_names):
                return get_moe_shard_specs(model, _fn, _names)

            self._workload.shard_spec_fn = combined_shard_spec_fn

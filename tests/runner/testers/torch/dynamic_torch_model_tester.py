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
from loguru import logger
from tt_torch.sparse_mlp import enable_sparse_mlp, get_moe_shard_specs

from tests.runner.test_utils import AdapterMode, RunPhase
from tests.runner.utils import TorchDynamicLoader
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.tools.lora import apply_lora_adapters


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
        test_metadata = None,
    ) -> None:
        """Initialize DynamicTorchModelTester.

        Args:
            run_mode: RunMode.INFERENCE or RunMode.TRAINING
            loader: Loader object that implements load_model and load_inputs methods
            comparison_config: Optional comparison configuration for result validation
            parallelism: Parallelism mode for model execution
            run_phase: Optional run phase (DEFAULT, LLM_DECODE, LLM_PREFILL)
            test_metadata: Optional ModelTestConfig with seq_len/batch_size for prefill
            and adapter_mode for LLM tests
        """
        # Create TorchDynamicLoader instance
        self.dynamic_loader = TorchDynamicLoader(loader)
        # Store parallelism for reporting/consumers
        self.parallelism = parallelism
        # Store phase hint for input loading
        self.run_phase = run_phase
        # Store test metadata for seq_len/batch_size/adapter_mode access
        self._test_metadata = test_metadata

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            compiler_config=compiler_config,
            run_mode=run_mode,
            parallelism=self.parallelism,
        )

        if test_metadata and getattr(test_metadata, "inject_custom_moe", False):
            self._inject_custom_moe(self._model)

    def _compile_for_tt_device(self, workload, options=None):
        """Apply per-variant weight dtype overrides before compiling for TT device."""
        self._apply_weight_dtype_overrides()
        super()._compile_for_tt_device(workload, options)
        self._remove_weight_dtype_overrides()

    def _apply_weight_dtype_overrides(self):
        """Auto-apply per-variant weight dtype overrides if available."""
        loader = self.dynamic_loader.loader
        if not hasattr(loader, "get_weight_dtype_config_path"):
            return
        try:
            config_path = loader.get_weight_dtype_config_path()
        except TypeError:
            return
        if config_path:
            from tt_torch.weight_dtype import apply_weight_dtype_overrides

            applied = apply_weight_dtype_overrides(self._model, config_path)
            if applied:
                logger.info(
                    f"Applied {len(applied)} weight dtype overrides from {config_path}"
                )

    def _remove_weight_dtype_overrides(self):
        """Remove weight dtype parametrizations after compilation.

        Parametrizations only need to be present during tracing/compilation to
        inject stablehlo custom_call metadata. Removing them afterwards prevents
        conflicts with tie_weights() during subsequent device placement.
        """
        from tt_torch.weight_dtype import remove_weight_dtype_overrides

        removed = remove_weight_dtype_overrides(self._model)
        if removed:
            logger.info(f"Removed {removed} weight dtype overrides after compilation")

    # --- TorchModelTester interface implementations ---

    def _get_model(self):
        """Get model instance from the dynamic loader.

        For LLM tests in training mode, adapters may be applied depending on
        adapter_mode (e.g. LORA reduces memory via low-rank matrices).

        Returns:
            Model instance loaded from the loader
        """
        model = self.dynamic_loader.load_model()

        adapter_mode = getattr(self._test_metadata, "adapter_mode", AdapterMode.NONE)
        if adapter_mode == AdapterMode.LORA and self._run_mode == RunMode.TRAINING:
            model = apply_lora_adapters(
                model,
                r=self._test_metadata.lora_r,
                lora_alpha=self._test_metadata.lora_alpha,
                target_modules=self._test_metadata.lora_target_modules,
                dropout=self._test_metadata.lora_dropout,
            )

        return model

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
            # TODO(umales): Remove inspect check, once we migrate to custom loader class for
            # Focus models.
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
                model, strategy=str(strategy), batch_axis=batch_axis
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
            # TODO(umales): Remove this once https://github.com/tenstorrent/tt-xla/issues/3397 is fixed
            if shard_inputs:
                mesh_names = ("data", "model")
            else:
                mesh_names = ("batch", "model")
        else:
            mesh_shape, mesh_names = self.dynamic_loader.get_mesh_config(num_devices)

        if mesh_shape and mesh_names:
            return get_mesh(mesh_shape, mesh_names)
        return None

    def _get_prefill_pcc_mask(self):
        """Return boolean token mask for LLM prefill PCC filtering."""
        if self.run_phase != RunPhase.LLM_PREFILL:
            return None

        attention_mask = self._input_activations.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            return None
        return attention_mask.to(dtype=torch.bool)

    def _compare(self, device_out, golden_out):
        """Compare outputs, masking padded prefill tokens for PCC only."""
        return self._evaluator.evaluate(
            device_out,
            golden_out,
            pcc_mask=self._get_prefill_pcc_mask(),
        )

    def _unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unwraps model output to a single tensor.
        Calls the unpack_forward_output method of the dynamic loader.
        """
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

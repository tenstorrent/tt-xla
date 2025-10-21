# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic JAX model tester implementation."""

import inspect
import jax
import jax.numpy as jnp
from infra.comparators import ComparisonConfig
from jax.sharding import Mesh

from third_party.tt_forge_models.config import Parallelism

from .dynamic_loader import JaxDynamicLoader
from .jax_model_tester import JaxModelTester
from .model_tester import RunMode


class DynamicJaxModelTester(JaxModelTester):
    """JAX model tester that uses a dynamic loader for model and input loading.

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
        """Initialize DynamicJaxModelTester.

        Args:
            run_mode: RunMode.INFERENCE or RunMode.TRAINING
            loader: Loader object that implements load_model and load_inputs methods
            comparison_config: Optional comparison configuration for result validation
            parallelism: Parallelism mode for model execution
        """
        # Create JaxDynamicLoader instance
        self.dynamic_loader = JaxDynamicLoader(loader)
        # Store parallelism for reporting/consumers
        self.parallelism = parallelism

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            run_mode=run_mode,
        )

    # --- JaxModelTester interface implementations ---

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
        return self.dynamic_loader.load_inputs()

    def _get_input_parameters(self):
        """Get input parameters from the dynamic loader if available.

        Returns:
            Input parameters from the loader if it implements get_input_parameters,
            otherwise delegates to parent implementation
        """
        # Check if loader has a specific method for input parameters
        if hasattr(self.dynamic_loader.loader, "get_input_parameters"):
            return self.dynamic_loader.loader.get_input_parameters()

        # Otherwise delegate to parent implementation which handles HF models
        return super()._get_input_parameters()

    def _get_shard_specs_function(self):
        """Get shard specs function from the dynamic loader if available.

        Returns:
            Shard spec function if loader supports it, None otherwise
        """
        return self.dynamic_loader.get_shard_spec_function()

    def _get_mesh(self):
        """Get mesh configuration from the dynamic loader if available.

        Returns:
            Mesh object if loader supports mesh configuration, None otherwise
        """
        # Get number of devices available
        num_devices = jax.device_count()
        mesh_shape, mesh_names = self.dynamic_loader.get_mesh_config(num_devices)

        if mesh_shape and mesh_names:
            # Create JAX mesh from the configuration
            devices = jax.devices()[:num_devices]
            devices_array = jnp.array(devices).reshape(mesh_shape)
            return Mesh(devices_array, mesh_names)
        return None

    def _get_forward_method_args(self):
        """Get forward method args, checking loader first then delegating to parent.

        Returns:
            Forward method args from loader if available, otherwise from parent
        """
        # Check if loader has specific forward method args
        if hasattr(self.dynamic_loader.loader, "get_forward_method_args"):
            return self.dynamic_loader.loader.get_forward_method_args()

        # Otherwise delegate to parent implementation
        return super()._get_forward_method_args()

    def _get_forward_method_kwargs(self):
        """Get forward method kwargs, checking loader first then delegating to parent.

        Returns:
            Forward method kwargs from loader if available, otherwise from parent
        """
        # Check if loader has specific forward method kwargs
        if hasattr(self.dynamic_loader.loader, "get_forward_method_kwargs"):
            # Pass run_mode if the method accepts it
            sig = inspect.signature(self.dynamic_loader.loader.get_forward_method_kwargs)
            if "run_mode" in sig.parameters:
                return self.dynamic_loader.loader.get_forward_method_kwargs(run_mode=self._run_mode)
            else:
                return self.dynamic_loader.loader.get_forward_method_kwargs()

        # Otherwise delegate to parent implementation
        return super()._get_forward_method_kwargs()

    def _get_static_argnames(self):
        """Get static argnames, checking loader first then delegating to parent.

        Returns:
            Static argnames from loader if available, otherwise from parent
        """
        # Check if loader has specific static argnames
        if hasattr(self.dynamic_loader.loader, "get_static_argnames"):
            return self.dynamic_loader.loader.get_static_argnames()

        # Otherwise delegate to parent implementation
        return super()._get_static_argnames()

    def _get_forward_method_name(self):
        """Get forward method name, checking loader first then using default.

        Returns:
            Forward method name from loader if available, otherwise "__call__"
        """
        # Check if loader has specific forward method name
        if hasattr(self.dynamic_loader.loader, "get_forward_method_name"):
            return self.dynamic_loader.loader.get_forward_method_name()

        # Default forward method for JAX models
        return "__call__"

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic JAX model tester implementation."""

import inspect

import jax
import jax.numpy as jnp
from infra.comparators import ComparisonConfig
from infra.testers.single_chip.model import JaxModelTester, RunMode

from tests.runner.utils import JaxDynamicLoader


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
    ) -> None:
        """Initialize DynamicJaxModelTester.

        Args:
            run_mode: RunMode.INFERENCE or RunMode.TRAINING
            loader: Loader object that implements load_model and load_inputs methods
            comparison_config: Optional comparison configuration for result validation
        """
        # Create JaxDynamicLoader instance
        self.dynamic_loader = JaxDynamicLoader(loader)

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

    def _get_forward_method_args(self):
        """Get forward method args, checking loader first then delegating to parent.

        Returns:
            Forward method args from loader if available, otherwise from parent
        """
        if hasattr(self.dynamic_loader.loader, "get_forward_method_args"):
            return self.dynamic_loader.loader.get_forward_method_args()

        return super()._get_forward_method_args()

    def _get_forward_method_kwargs(self):
        """Get forward method kwargs, checking loader first then delegating to parent.

        Returns:
            Forward method kwargs from loader if available, otherwise from parent
        """
        # Check if loader has specific forward method kwargs
        if hasattr(self.dynamic_loader.loader, "get_forward_method_kwargs"):
            # Pass run_mode if the method accepts it
            sig = inspect.signature(
                self.dynamic_loader.loader.get_forward_method_kwargs
            )
            if "run_mode" in sig.parameters:
                return self.dynamic_loader.loader.get_forward_method_kwargs(
                    run_mode=self._run_mode
                )
            else:
                return self.dynamic_loader.loader.get_forward_method_kwargs()

        # Otherwise delegate to parent implementation
        return super()._get_forward_method_kwargs()

    def _get_static_argnames(self):
        """Get static argnames, checking loader first then delegating to parent.

        Returns:
            Static argnames from loader if available, otherwise from parent
        """
        if hasattr(self.dynamic_loader.loader, "get_static_argnames"):
            return self.dynamic_loader.loader.get_static_argnames()

        return super()._get_static_argnames()

    def _get_forward_method_name(self):
        """Get forward method name, checking loader first then using default.

        Returns:
            Forward method name from loader if available, otherwise "__call__"
        """
        if hasattr(self.dynamic_loader.loader, "get_forward_method_name"):
            return self.dynamic_loader.loader.get_forward_method_name()

        return "__call__"

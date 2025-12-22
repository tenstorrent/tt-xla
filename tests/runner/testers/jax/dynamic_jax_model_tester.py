# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic JAX model tester implementation."""

import inspect

import jax
from flax import linen
from infra.comparators import ComparisonConfig
from infra.testers.single_chip.model import JaxModelTester, RunMode
from transformers import FlaxPreTrainedModel

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
            # Pass train flag if the method accepts it
            sig = inspect.signature(self.dynamic_loader.loader.get_input_parameters)
            if "train" in sig.parameters:
                train = self._run_mode == RunMode.TRAINING
                return self.dynamic_loader.loader.get_input_parameters(train=train)
            else:
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
        # First get the base kwargs from parent (includes params and inputs for HF models)
        kwargs = super()._get_forward_method_kwargs()

        # Check if loader has specific forward method kwargs to add/override
        if hasattr(self.dynamic_loader.loader, "get_forward_method_kwargs"):
            # Pass train flag if the method accepts it
            sig = inspect.signature(
                self.dynamic_loader.loader.get_forward_method_kwargs
            )
            if "train" in sig.parameters:
                # Pass train=True if we're in training mode
                train = self._run_mode == RunMode.TRAINING
                loader_kwargs = self.dynamic_loader.loader.get_forward_method_kwargs(
                    train=train
                )
            else:
                loader_kwargs = self.dynamic_loader.loader.get_forward_method_kwargs()

            # For HuggingFace models, merge the loader kwargs with parent kwargs
            # For custom models, the loader kwargs replace the parent kwargs
            model = self._get_model()
            if isinstance(model, FlaxPreTrainedModel):
                # HuggingFace model: merge (loader provides RNG keys, parent provides inputs/params)
                kwargs.update(loader_kwargs)
            else:
                # Custom model: replace (loader provides everything)
                kwargs = loader_kwargs

        # Add dropout_rng for HuggingFace models in training mode (if not already present)
        # This handles HuggingFace models that don't have a custom loader
        # Only add if the model actually accepts dropout_rng parameter
        if self._run_mode == RunMode.TRAINING:
            model = self._get_model()
            if isinstance(model, FlaxPreTrainedModel):
                if "dropout_rng" not in kwargs:
                    # Check if the model's __call__ method accepts dropout_rng
                    sig = inspect.signature(model.__call__)
                    if "dropout_rng" in sig.parameters:
                        kwargs["dropout_rng"] = jax.random.key(1)

        return kwargs

    def _get_static_argnames(self):
        """Get static argnames, checking loader first then delegating to parent.

        Returns:
            Static argnames from loader if available, otherwise from parent
        """
        if hasattr(self.dynamic_loader.loader, "get_static_argnames"):
            loader_static_args = self.dynamic_loader.loader.get_static_argnames()
            # For HuggingFace models, also check parent's logic for train/deterministic
            model = self._get_model()
            if isinstance(model, FlaxPreTrainedModel):
                # Get parent's static argnames (which checks model signature for train/deterministic)
                parent_static_args = super()._get_static_argnames()
                # Merge both lists, keeping unique values
                return list(set(loader_static_args + parent_static_args))
            return loader_static_args

        return super()._get_static_argnames()

    def _get_forward_method_name(self):
        """Get forward method name, checking loader first then using default.

        Returns:
            Forward method name from loader if available, "apply" for Flax
            linen.Module models, otherwise "__call__"
        """
        if hasattr(self.dynamic_loader.loader, "get_forward_method_name"):
            return self.dynamic_loader.loader.get_forward_method_name()
        elif isinstance(self._get_model(), linen.Module):
            return "apply"

        return "__call__"

    def _wrapper_model(self, f):
        """Wrapper for model forward method that extracts the appropriate output.

        First checks if the loader provides a custom wrapper_model method.
        If not, delegates to parent implementation.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function from loader or parent implementation
        """
        # Check if loader provides a custom wrapper
        if hasattr(self.dynamic_loader.loader, "wrapper_model"):
            return self.dynamic_loader.loader.wrapper_model(f)

        # Otherwise delegate to parent implementation
        return super()._wrapper_model(f)

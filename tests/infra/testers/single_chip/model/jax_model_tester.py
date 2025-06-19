# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from flax import linen, nnx
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from utilities.types import Model
from utilities.workloads.jax_workload import JaxWorkload, Workload

from .model_tester import ModelTester


class JaxModelTester(ModelTester):
    """
    Single chip `jax` model tester.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations(self) -> Sequence[Any]
    _get_forward_method_name(self) -> str # Optional, has default behaviour.
    _get_static_argnames(self) -> Sequence[str] # Optional, has default behaviour.
    # One of or both:
    _get_forward_method_args(self) -> Sequence[Any] # Optional, has default behaviour.
    _get_forward_method_kwargs(self) -> Mapping[str, Any] # Optional, has default behaviour.
    ```
    """

    # -------------------- Private methods --------------------

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        """
        Return the names of arguments which should be treated as static by JIT compiler.
        Static arguments are those which are not replaced with Tracer objects by the JIT
        but rather are used as is, which is needed if control flow or shapes depend on them.
        https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables


        By default no arguments are static.
        """
        return []

    # --- Overrides ---

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model's forward pass into optimized kernels."""
        assert isinstance(workload, JaxWorkload)

        workload.executable = jax.jit(
            workload.executable, static_argnames=workload.static_argnames
        )
        return workload

    # @override
    def _configure_model_for_inference(self, model: Model) -> None:
        assert isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(model, nnx.Module):
            # TODO find another way to do this since model.eval() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        model.eval()

    # @override
    def _configure_model_for_training(self, model: Model) -> None:
        assert isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(model, nnx.Module):
            # TODO find another way to do this since model.train() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        model.train()

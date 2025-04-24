# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
from flax import linen, nnx
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from utilities.types import Model
from utilities.workloads.jax_workload import JaxWorkload, Workload

from .framework_adapter import FrameworkAdapter


class JaxAdapter(FrameworkAdapter):
    """Adapter for JAX."""

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _configure_model_for_inference(self, model: Model) -> None:
        if isinstance(model, nnx.Module):
            model.eval()
        elif isinstance(model, (linen.Module, FlaxPreTrainedModel)):
            # TODO find another way to do this since model.eval() does not exist, maybe
            # by passing train param as kwarg to __call__.
            pass
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

    # @override
    def _configure_model_for_training(self, model: Model) -> None:
        if isinstance(model, nnx.Module):
            model.train()
        elif isinstance(model, (linen.Module, FlaxPreTrainedModel)):
            # TODO find another way to do this since model.train() does not exist, maybe
            # by passing train param as kwarg to __call__.
            pass
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model's forward pass into optimized kernels."""
        assert isinstance(workload, JaxWorkload)

        workload.executable = jax.jit(
            workload.executable, static_argnames=workload.static_argnames
        )
        return workload

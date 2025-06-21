# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from utilities.types import Framework, Model
from utilities.workloads import Workload, WorkloadFactory
from utilities.workloads.torch_workload import TorchWorkload, Workload

from .model_tester import ModelTester


class TorchModelTester(ModelTester):
    """
    Single chip `torch` model tester.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations(self) -> Sequence[Any]
    _get_forward_method_name(self) -> str # Optional, has default behaviour.
    # One of or both:
    _get_forward_method_args(self) -> Sequence[Any] # Optional, has default behaviour.
    _get_forward_method_kwargs(self) -> Mapping[str, Any] # Optional, has default behaviour.
    ```
    """

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _initialize_framework(self) -> None:
        self._framework = Framework.TORCH

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"

        self._workload = WorkloadFactory(self._framework).create_workload(
            model=self._model, args=args, kwargs=kwargs
        )

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations = self._get_input_activations()

    # @override
    def _configure_model_for_inference(self, model: Model) -> None:
        assert isinstance(model, torch.nn.Module)
        model.eval()

    # @override
    def _configure_model_for_training(self, model: Model) -> None:
        assert isinstance(model, torch.nn.Module)
        model.train()

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model into optimized kernels."""
        assert isinstance(workload, TorchWorkload)
        assert workload.model is not None

        workload.model.compile(backend="openxla")
        return workload

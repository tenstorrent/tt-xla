# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import torch
from infra.comparators import ComparisonConfig
from infra.utilities import Framework, Model
from infra.workloads import TorchWorkload, Workload, WorkloadFactory

from .model_tester import ModelTester, RunMode


class TorchModelTester(ModelTester):
    """
    Abstract base class all single chip `torch` model testers must inherit.

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

    # -------------------- Protected methods --------------------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        # Placeholders for objects that will be set during
        # `_initialize_all_components`. Easier to spot if located in constructor instead
        # of dynamically creating them somewhere in methods.
        self._input_activations: Dict | Sequence[Any] = None

        super().__init__(comparison_config, run_mode, Framework.TORCH)

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"

        self._workload = WorkloadFactory.create_workload(
            self._framework, model=self._model, args=args, kwargs=kwargs
        )

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations = self._get_input_activations()

    # @override
    @staticmethod
    def _configure_model_for_inference(model: Model) -> None:
        assert isinstance(model, torch.nn.Module)
        model.eval()

    # @override
    @staticmethod
    def _configure_model_for_training(model: Model) -> None:
        assert isinstance(model, torch.nn.Module)
        model.train()

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model into optimized kernels."""
        assert isinstance(workload, TorchWorkload) and workload.model is not None

        workload.model.compile(backend="openxla")
        return workload

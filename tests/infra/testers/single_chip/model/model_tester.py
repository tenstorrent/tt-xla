# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

from infra.comparators import ComparisonConfig
from infra.utilities import Framework, Model, Tensor
from infra.workloads import Workload

from ...base_tester import BaseTester


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


class ModelTester(BaseTester, ABC):
    """Abstract base class all single chip model testers must inherit."""

    def __init__(
        self,
        comparison_config: ComparisonConfig,
        run_mode: RunMode,
        framework: Framework,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._run_mode = run_mode

        self._model: Model = None
        self._workload: Workload = None

        super().__init__(comparison_config, framework)
        self._initialize_all_components()
        
    def _initialize_all_components(self) -> None:
        self._initialize_model()
        self._initialize_workload()
        
    def _initialize_model(self) -> None:
        """
        Initializes `self._model`

        It is also important that model is configured before it is prepacked into a
        Workload during `_initialize_workload`.
        """
        # Store model instance.
        self._model = self._get_model()
        # Configure it.
        self._configure_model()
        # Cache model inputs.
        self._cache_model_inputs()

    @abstractmethod
    def _get_model(self) -> Model:
        """Returns model instance."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _configure_model(self) -> None:
        """
        Configures model for inference *or* training, depending on chosen run mode.
        """
        if self._run_mode == RunMode.INFERENCE:
            self._configure_model_for_inference()
        else:
            self._configure_model_for_training()

    @staticmethod
    @abstractmethod
    def _configure_model_for_inference(model: Model) -> None:
        """Configures `model` for inference."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _configure_model_for_training(model: Model) -> None:
        """Configures `model` for training."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_forward_method_name(self) -> str:
        """
        Returns string name of model's forward pass method.

        Returns "__call__" by default which is the most common one. "forward" and
        "apply" are also common.
        """
        return "__call__"

    def test(self) -> None:
        """Tests the model depending on test type with which tester was configured."""
        if self._run_mode == RunMode.INFERENCE:
            self._test_inference()
        else:
            self._test_training()

    def _test_inference(self) -> None:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        compiled_cpu_workload = self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(compiled_cpu_workload)

        compiled_device_workload = self._compile_for_tt_device(self._workload)
        tt_res = self._run_on_tt_device(compiled_device_workload)

        self._compare(tt_res, cpu_res)

    @abstractmethod
    def _compile_for_cpu(self, workload: Workload) -> Workload:
        """Compiles `workload` for CPU."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._device_runner.run_on_cpu(compiled_workload)

    @abstractmethod
    def _compile_for_tt_device(self, workload: Workload) -> Workload:
        """Compiles `workload` for TT device."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> None:
        """Compares device with golden output."""
        self._comparator.compare(device_out, golden_out)

    def _test_training(self):
        """TODO"""
        raise NotImplementedError("Support for training not implemented")

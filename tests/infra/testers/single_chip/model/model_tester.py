# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

from comparators import ComparisonConfig
from testers.base_tester import BaseTester
from utilities.types import Model
from utilities.workloads import Workload


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


class ModelTester(BaseTester, ABC):
    """Abstract base class all single chip model testers must inherit."""

    # -------------------- Public methods --------------------

    def test(self) -> None:
        """Tests the model depending on test type with which tester was configured."""
        if self._run_mode == RunMode.INFERENCE:
            self._test_inference()
        else:
            self._test_training()

    # ---------- Protected methods ----------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._run_mode = run_mode
        # Placeholders for objects that will be set in `_initialize_all_components`.
        # Easier to spot if located in constructor instead of dynamically creating them
        # somewhere in methods.
        self._model: Model = None
        self._workload: Workload = None

        # Framework not passed for now, determined in `_initialize_framework`.
        super().__init__(comparison_config)

    # --- For test writer's tester subclasses to override ---

    @abstractmethod
    def _get_model(self) -> Model:
        """Returns model instance."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        """Returns input activations."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_forward_method_name(self) -> str:
        """
        Returns string name of model's forward pass method.

        Returns "__call__" by default which is the most common one. "forward" and
        "apply" are also common.
        """
        return "__call__"

    def _get_forward_method_args(self) -> Sequence[Any]:
        """
        Returns positional arguments for model's forward pass.

        By default returns empty list.
        """
        return []

    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Returns keyword arguments for model's forward pass.

        By default returns empty dict.
        """
        return {}

    # --- These are overridden by framework-specific child classes ---

    @abstractmethod
    def _initialize_framework(self) -> None:
        """Initializes `self._model` and `self._framework`."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _configure_model_for_inference(self, model: Model) -> None:
        """Configures `model` for inference."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _configure_model_for_training(self, model: Model) -> None:
        """Configures `model` for training."""
        raise NotImplementedError("Subclasses must implement this method")

    # -------------------- Private methods --------------------

    # --- Tester initialization ---

    # @override
    def _initialize_all_components(self) -> None:
        """
        Helper initialization method which handles validation of provided interface
        methods concrete tester subclasses must implement and sets some useful
        attributes.

        It stores framework in use, based on the framework instantiates a DeviceRunner
        (which internally instantiates a DeviceConnector singleton, ensuring plugin
        registration and connection to the device) and a Comparator, instantiates and
        stores the concrete model instance and finally packs model or its forward method
        and its arguments in a Workload.

        NOTE It is important that plugin registration is conducted before any other
        framework-specific commands are executed. For example, getting forward method
        args or kwargs must certainly invoke `_get_input_activations` which might
        contain some framework specific commands (like jax.ones(...)). So we should
        strive to create a DeviceRunner as soon as possible. Only after determining
        framework are we able to instatiate a proper DeviceRunner (i.e. establish device
        connection).
        """
        self._initialize_framework()
        self._initialize_framework_specific_helpers()
        self._initialize_model()
        self._initialize_workload()

    def _initialize_model(self) -> None:
        """Initializes `self._model`"""
        # Store model instance.
        self._model = self._get_model()
        # Cache model inputs.
        self._cache_model_inputs()

    # --- Testing methods ---

    def _test_inference(self) -> None:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        self._configure_model_for_inference()

        compiled_workload = self._compile(self._workload)

        tt_res = self._run_on_tt_device(compiled_workload)
        cpu_res = self._run_on_cpu(compiled_workload)

        self._compare(tt_res, cpu_res)

    def _test_training(self):
        """TODO"""
        raise NotImplementedError("Support for training not implemented")

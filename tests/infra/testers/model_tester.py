# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Union

from adapters import FrameworkAdapter, FrameworkAdapterFactory
from comparators import Comparator, ComparatorFactory, ComparisonConfig
from runners import DeviceRunner, DeviceRunnerFactory
from utilities.types import Framework, Model, Tensor
from utilities.workloads import Workload, WorkloadFactory

from .base_tester import BaseTester


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


class ModelTester(BaseTester, ABC):
    """
    Abstract base class all model testers must inherit.

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

    # -------------------- Public methods --------------------

    def test(self) -> None:
        """Tests the model depending on test type with which tester was configured."""
        if self._run_mode == RunMode.INFERENCE:
            self._test_inference()
        else:
            self._test_training()

    # -------------------- Protected methods --------------------

    # --- For subclasses to override ---

    @abstractmethod
    def _get_model(self) -> Model:
        """Returns model instance."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_input_activations(self) -> Union[Dict, Sequence[Any]]:
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

        `self` is provided for convenience, for example if some model attribute needs
        to be fetched.
        """
        return []

    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Returns keyword arguments for model's forward pass.

        By default returns empty dict.

        `self` is provided for convenience, for example if some model attribute needs
        to be fetched.
        """
        return {}

    def _get_static_argnames(self) -> Sequence[str]:
        """
        Return the names of arguments which should be treated as static by JIT compiler.
        Static arguments are those which are not replaced with Tracer objects by the JIT
        but rather are used as is, which is needed if control flow or shapes depend on them.
        https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables


        By default no arguments are static.
        """
        return []

    # ----------------------------------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._comparison_config = comparison_config
        self._run_mode = run_mode
        # Placeholders for objects that will be set in `_initialize_all_components`.
        # Easier to spot if located in constructor instead of dynamically creating them
        # somewhere in methods.
        self._model: Model = None
        self._framework: Framework = None
        self._device_runner: DeviceRunner = None
        self._adapter: FrameworkAdapter = None
        self._comparator: Comparator = None
        self._workload: Workload = None

        self._initialize_all_components()

    # -------------------- Private methods --------------------

    # --- Tester initialization ---

    def _initialize_all_components(self) -> None:
        """
        Helper initialization method which handles validation of provided interface
        methods concrete tester subclasses must implement and sets some useful
        attributes.

        It instantiates and stores the concrete model instance, determines in which
        framework it was written, based on the framework instantiates a DeviceRunner
        (which internally instantiates a DeviceConnector singleton, ensuring plugin
        registration and connection to the device), a FrameworkAdapter and a Comparator,
        and finally packs model's forward method and its arguments in a Workload.

        NOTE It is important that plugin registration is conducted before any other
        framework-specific commands are executed. For example, getting forward method
        args or kwargs must certainly invoke `_get_input_activations` which might
        contain some framework specific commands (like jax.ones(...)). So we should
        strive to create a DeviceRunner as soon as possible. To avoid forcing user to
        explicitly pass a framework, it is automatically inferred from the model. Only
        then we are able to instatiate a proper DeviceRunner (i.e. establish device
        connection).
        """
        self._initialize_model_and_framework()
        self._initialize_framework_specific_helpers()
        self._initialize_workload()

    def _initialize_model_and_framework(self) -> None:
        """Initializes `self._model` and `self._framework`."""
        # Store model instance.
        self._model = self._get_model()
        # Determine framework in which the model was written.
        self._framework = Framework.from_model_type(self._model)

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner`, `self._adapter` and `self._comparator`.

        This function triggers connection to device.
        """
        # Creating runner will register plugin and connect the device properly.
        self._device_runner: DeviceRunner = DeviceRunnerFactory(
            self._framework
        ).create_runner()
        self._adapter: FrameworkAdapter = FrameworkAdapterFactory(
            self._framework
        ).create_adapter()
        self._comparator: Comparator = ComparatorFactory(
            self._framework
        ).create_comparator(self._comparison_config)

    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()
        forward_static_args = self._get_static_argnames()
        forward_method_name = self._get_forward_method_name()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"
        assert hasattr(
            self._model, forward_method_name
        ), f"Model does not have {forward_method_name} method provided."

        forward_pass_method = getattr(self._model, forward_method_name)

        self._workload = WorkloadFactory(self._framework).create_workload(
            executable=forward_pass_method,
            model=self._model,
            args=args,
            kwargs=kwargs,
            static_argnames=forward_static_args,
        )

    # --- Testing methods ---

    def _test_inference(self) -> None:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        self._configure_model_for_inference()

        compiled_workload = self._compile()

        tt_res = self._run_on_tt_device(compiled_workload)
        cpu_res = self._run_on_cpu(compiled_workload)

        self._compare(tt_res, cpu_res)

    def _test_training(self):
        """TODO"""
        raise NotImplementedError("Support for training not implemented")

    # --- Private convenience wrappers ---

    def _configure_model_for_inference(self) -> None:
        """Configures model for inference."""
        self._adapter.configure_model_for_inference(self._model)

    def _configure_model_for_training(self) -> None:
        """Configures model for training."""
        self._adapter.configure_model_for_training(self._model)

    def _compile(self) -> Workload:
        """
        Compiles workload into optimized kernels.

        Returns new "compiled" Workload.
        """
        return self._adapter.compile(self._workload)

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._device_runner.run_on_cpu(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> None:
        """Compares device with golden output."""
        self._comparator.compare(device_out, golden_out)

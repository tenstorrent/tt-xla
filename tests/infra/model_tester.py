# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Sequence, Union

from flax import linen, nnx
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from .base_tester import BaseTester
from .comparison import ComparisonConfig
from .device_runner import DeviceRunner
from .types import Model
from .workload import Workload


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"


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

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(comparison_config)

        self._run_mode = run_mode

        self._init_model_hooks()

    def _init_model_hooks(self) -> None:
        """
        Extracted init method which handles validation of provided interface methods
        subclasses must implement and storing of some useful return values.
        """
        # Store model instance.
        self._model = self._get_model()

        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()

        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError(
                f"Forward method args or kwargs or both must be provided"
            )

        forward_method_name = self._get_forward_method_name()

        if not hasattr(self._model, forward_method_name):
            raise ValueError(
                f"Model does not have {forward_method_name} method provided."
            )

        forward_pass_method = getattr(self._model, forward_method_name)

        forward_static_args = self._get_static_argnames()

        # Store model's forward pass method and its arguments as a workload.
        self._workload = Workload(
            forward_pass_method, args, kwargs, forward_static_args
        )

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
        ModelTester._configure_model_for_inference(self._model)

        compiled_forward_method = self._compile_model()

        compiled_workload = Workload(
            compiled_forward_method,
            self._workload.args,
            self._workload.kwargs,
            self._workload.static_argnames,
        )

        tt_res = DeviceRunner.run_on_tt_device(compiled_workload)
        cpu_res = DeviceRunner.run_on_cpu(compiled_workload)

        self._compare(tt_res, cpu_res)

    def _test_training(self):
        """TODO"""
        # self._configure_model_for_training(model)
        raise NotImplementedError("Support for training not implemented")

    @staticmethod
    def _configure_model_for_inference(model: Model) -> None:
        """Configures model for inference."""
        if isinstance(model, nnx.Module):
            model.eval()
        elif isinstance(model, (linen.Module, FlaxPreTrainedModel)):
            # TODO find another way to do this since model.eval() does not exist, maybe
            # by passing train param as kwarg to __call__.
            pass
        else:
            raise TypeError(f"Uknown model type: {type(model)}")

    @staticmethod
    def _configure_model_for_training(model: Model) -> None:
        """Configures model for training."""
        if isinstance(model, nnx.Module):
            model.train()
        elif isinstance(model, (linen.Module, FlaxPreTrainedModel)):
            # TODO find another way to do this since model.train() does not exist, maybe
            # by passing train param as kwarg to __call__.
            pass
        else:
            raise TypeError(f"Uknown model type: {type(model)}")

    def _compile_model(self) -> Callable:
        """JIT-compiles model's forward pass into optimized kernels."""
        return super()._compile(
            self._workload.executable, self._workload.static_argnames
        )

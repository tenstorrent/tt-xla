# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Callable, Sequence, Union

import jax

from .device_runner import DeviceRunner
from .test_model import TestModel
from .test_module import TestModule
from .utils import (
    compare_allclose,
    compare_atol,
    compare_equal,
    compare_pcc,
    random_tensor,
)


class TestType(Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class ComparisonMetric(Enum):
    EQUAL = "equal"
    PCC = "pcc"
    ATOL = "atol"
    ALLCLOSE = "allclose"


class ModuleTester:
    """
    Class providing infrastructure for testing test modules.

    Testing consists of comparing output results of running provided test module on
    different devices. Frequently, run on CPU or GPU is taken as a ground truth
    ("golden"), while custom device run is compared to it.

    It supports testing inference and training modes.
    """

    def __init__(
        self,
        test_type: TestType = TestType.INFERENCE,
        comparison_metric: ComparisonMetric = ComparisonMetric.PCC,
    ) -> None:
        self._test_type = test_type
        self._comparison_metric = comparison_metric

    def __call__(self, module: TestModule) -> bool:
        """
        The only public method providing testing hook.

        Call tester with passed test module and it will run tests on it.
        """
        return self._test(module)

    def _test(self, module: TestModule) -> bool:
        if self._test_type == TestType.INFERENCE:
            return self._test_inference(module)
        else:
            return self._test_training(module)

    def _test_inference(self, module: TestModule) -> bool:
        tt_res = DeviceRunner.run_on_tt_device(module)
        cpu_res = DeviceRunner.run_on_cpu(module)
        return self._compare(tt_res, cpu_res)

    def _test_training(self, module: TestModule) -> bool:
        raise NotImplementedError("Support for training not implemented")

    def _compare(
        self, device_out: jax.Array, golden_out: jax.Array, assert_on_fail: bool = False
    ) -> bool:
        device_output, golden_output = DeviceRunner.put_on_cpu(device_out, golden_out)
        device_output, golden_output = self._match_data_types(
            device_output, golden_output
        )

        if self._comparison_metric == ComparisonMetric.EQUAL:
            comp = compare_equal(device_output, golden_output)
        elif self._comparison_metric == ComparisonMetric.PCC:
            comp = compare_pcc(device_output, golden_output)
        elif self._comparison_metric == ComparisonMetric.ATOL:
            comp = compare_atol(device_output, golden_output)
        elif self._comparison_metric == ComparisonMetric.ALLCLOSE:
            comp = compare_allclose(device_output, golden_output)

        if assert_on_fail:
            assert comp, f"{self._comparison_metric.value} comparison failed!"

        return comp

    def _match_data_types(self, *tensors: jax.Array) -> Sequence[jax.Array]:
        """
        Casts all tensors to float32 if not already in that format.

        Tensors need to be in same data format in order to compare them.
        """
        return [
            tensor.astype("float32") if tensor.dtype.str != "float32" else tensor
            for tensor in tensors
        ]


def _test(
    f: Union[Callable, TestModel],
    inputs: Sequence[jax.Array],
    comparison_metric: ComparisonMetric,
) -> bool:
    """Helper 'protected' method, don't use, use provided public methods below instead."""
    tester = ModuleTester(comparison_metric=comparison_metric)
    module = (
        f.as_test_module(inputs)
        if isinstance(f, TestModel)
        else TestModule(f, args=inputs)
    )
    return tester(module)


def test(
    f: Union[Callable, TestModel],
    inputs: Sequence[jax.Array],
    comparison_metric: ComparisonMetric,
) -> bool:
    return _test(f, inputs, comparison_metric)


def test_with_random_inputs(
    f: Union[Callable, TestModel],
    input_shapes: Sequence[tuple],
    comparison_metric: ComparisonMetric,
) -> bool:
    inputs = [random_tensor(shape) for shape in input_shapes]
    return _test(f, inputs, comparison_metric)


# TODO expose multiple functions for each of the comparisons since their args may differ.

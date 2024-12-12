# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import Callable, Sequence

import jax

from .comparison import (
    ComparisonConfig,
    compare_allclose,
    compare_atol,
    compare_equal,
    compare_pcc,
)
from .device_runner import DeviceRunner
from .utils import Model, Tensor, random_tensor

# --------------------------------------------------------------------------------------


class BaseTester(ABC):
    def __init__(
        self, comparison_config: ComparisonConfig = ComparisonConfig()
    ) -> None:
        self._comparison_config = comparison_config

    def _compare(
        self,
        device_out: Tensor,
        golden_out: Tensor,
    ) -> None:
        device_output, golden_output = DeviceRunner.put_on_cpu(device_out, golden_out)
        device_output, golden_output = self._match_data_types(
            device_output, golden_output
        )

        if self._comparison_config.equal.enabled:
            assert compare_equal(
                device_output, golden_output
            ), f"Equal comparison failed"

        if self._comparison_config.atol.enabled:
            assert compare_atol(
                device_output, golden_output, self._comparison_config.atol
            ), f"Atol comparison failed"

        if self._comparison_config.pcc.enabled:
            assert compare_pcc(
                device_output, golden_output, self._comparison_config.pcc
            ), f"PCC comparison failed"

        if self._comparison_config.allclose.enabled:
            assert compare_allclose(
                device_output, golden_output, self._comparison_config.allclose
            ), f"Allclose comparison failed"

    def _match_data_types(self, *tensors: Tensor) -> Sequence[Tensor]:
        """
        Casts all tensors to float32 if not already in that format.

        Tensors need to be in same data format in order to compare them.
        """
        return [
            (tensor.astype("float32") if tensor.dtype.str != "float32" else tensor)
            for tensor in tensors
        ]

    @staticmethod
    def _compile(f: Callable) -> Callable:
        return jax.jit(f.__call__)


# --------------------------------------------------------------------------------------


class OpTester(BaseTester):
    def __init__(self, comparison_metric: ComparisonConfig) -> None:
        super().__init__(comparison_metric)

    def test(self, f: Callable, inputs: Sequence[Tensor]) -> None:
        return self._test(f, inputs)

    def test_with_random_inputs(
        self, f: Callable, input_shapes: Sequence[tuple]
    ) -> None:
        inputs = [random_tensor(shape) for shape in input_shapes]
        return self._test(f, inputs)

    def _test(self, f: Callable, inputs: Sequence[Tensor]) -> None:
        # TODO compile
        tt_res = DeviceRunner.run_on_tt_device(f, inputs)
        cpu_res = DeviceRunner.run_on_cpu(f, inputs)
        return self._compare(tt_res, cpu_res)


def run_op_test(
    op: Callable, inputs: Sequence[Tensor], comparison_metric: ComparisonConfig
) -> None:
    tester = OpTester(comparison_metric)
    tester.test(op, inputs)


def run_op_test_with_random_inputs(
    op: Callable, input_shapes: Sequence[tuple], comparison_metric: ComparisonConfig
) -> None:
    tester = OpTester(comparison_metric)
    tester.test_with_random_inputs(op, input_shapes)


# --------------------------------------------------------------------------------------


class GraphTester(OpTester):

    pass


def run_graph_test(
    graph: Callable, inputs: Sequence[Tensor], comparison_metric: ComparisonConfig
) -> None:
    tester = GraphTester(comparison_metric)
    tester.test(graph, inputs)


def run_graph_test_with_random_inputs(
    graph: Callable, input_shapes: Sequence[tuple], comparison_metric: ComparisonConfig
) -> None:
    tester = GraphTester(comparison_metric)
    tester.test_with_random_inputs(graph, input_shapes)


# --------------------------------------------------------------------------------------


class TestType(Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class ModelTester(BaseTester):
    def __init__(
        self,
        comparison_metric: ComparisonConfig,
        test_type: TestType = TestType.INFERENCE,
    ) -> None:
        super().__init__(comparison_metric)
        self._test_type = test_type

    def load_model() -> Model:
        pass

    def load_inputs() -> Sequence[Tensor]:
        pass

    def compile_model():
        pass

    def test(self, model: Model) -> bool:
        if self._test_type == TestType.INFERENCE:
            return self._test_inference(model)
        else:
            return self._test_training(model)

    def _test_inference(self, model: Model):
        tt_res = DeviceRunner.run_on_tt_device(model)
        cpu_res = DeviceRunner.run_on_cpu(model)
        return self._compare(tt_res, cpu_res)

    def _test_training():
        raise NotImplementedError("Support for training not implemented")

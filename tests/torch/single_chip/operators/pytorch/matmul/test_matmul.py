# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch

from typing import List
from loguru import logger

from ...utils import (
    FailingReasons,
    InputSource,
    PytorchUtils,
    ShapeUtils,
    TensorUtils,
    TestCollection,
    TestCollectionCommon,
    TestCollectionTorch,
    TestDevice,
    TestPlan,
    TestVector,
    ValueCheckerUtils,
    ValueRanges,
    VerifyConfig,
    VerifyUtils,
)
from ..ids import TestIdsDataLoader


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_another_op"
        self.operator = operator

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, yy)


class ModelFromHost(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_host"
        self.operator = operator

    def forward(self, x, y):
        return self.operator(x, y)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape_1, shape_2, dtype, value_range):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.c1 = TensorUtils.create_torch_constant(
            input_shape=shape_1,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(shape_1),
        )
        self.c2 = TensorUtils.create_torch_constant(
            input_shape=shape_2,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=sum(shape_2),
        )
        self.register_buffer("constant1", self.c1)
        self.register_buffer("constant2", self.c2)

    def forward(self, x, y):
        mm1 = self.operator(self.c1, self.c2)
        mm2 = self.operator(x, y)
        aa = torch.add(mm1, mm2)
        return aa


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelFromHost,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        warm_reset: bool = False,
    ):
        model_type = cls.MODEL_TYPES[test_vector.input_source]

        value_range = ValueRanges.SMALL
        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)

        if isinstance(test_vector.input_shape[0], int):
            # input_shape is tuple of ints e.g. (a, b)
            input_shapes = [test_vector.input_shape, ShapeUtils.switch_last_two(test_vector.input_shape)]
        else:
            # input_shape is tuple of tuples e.g. ((a, b), (c, d))
            input_shapes = [test_vector.input_shape[0], test_vector.input_shape[1]]

        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator, *input_shapes, dtype=test_vector.dev_data_format, value_range=value_range
            )
        else:
            pytorch_model = model_type(operator)

        logger.trace(f"***input_shapes: {input_shapes}")

        value_checker = ValueCheckerUtils.automatic(pcc=0.99, rtol=1e-2, atol=1e-2)

        verify_config = VerifyConfig(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_range=value_range,
            value_checker=value_checker,
        )
        VerifyUtils.verify(verify_config)


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operators = ["matmul"]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test all shapes and input sources collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # Test special cases collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                ((3,), (3,)),
                ((7, 3), (3, 7)),
                ((3,), (3, 7)),
                ((32, 64), (64,)),
                ((64,), (3, 1, 64, 32)),
            ],
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operators),
        # Memory issue (too large input shapes):
        TestCollection(
            operators=TestParamsData.operators,
            input_shapes=[(10, 10, 10000, 1)],
            skip_reason=FailingReasons.OUT_OF_MEMORY,
            failing_reason=FailingReasons.OUT_OF_MEMORY,
        ),
        # # Unsupported data format:
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_sources=TestCollectionCommon.single.input_sources,
        #     input_shapes=TestCollectionCommon.single.input_shapes,
        #     dev_data_formats=[
        #         torch.int8,
        #         torch.int32,
        #         torch.int64,
        #         torch.float16,
        #     ],
        #     failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        # ),
        # # Matmul op when two input tensors are vectors is not supported:
        # #    >> tvm.error.InternalError
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_shapes=[((3,), (3,))],
        #     failing_reason=FailingReasons.COMPILATION_FAILED,
        # ),
        # # Matmul op when one of the arguments is 1-dimensional and the other one is N-dimensional:
        # #    >> AssertionError: Data mismatch on output 0 between framework and Forge codegen
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_shapes=[((64,), (3, 1, 64, 32))],
        #     failing_reason=FailingReasons.DATA_MISMATCH,
        # ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]

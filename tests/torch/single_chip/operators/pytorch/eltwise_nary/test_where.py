# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import math
import torch

from typing import List
from loguru import logger

from ...utils import (
    InputSource,
    PytorchUtils,
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
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "where_operator_test_op_src_from_another_op"
        self.operator = operator
        self.treshold = 0.3

    def forward(self, x, y, z):
        condition = x > self.treshold
        input = torch.add(y, y)
        other = torch.add(z, z)
        return self.operator(condition, input, other)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator):
        super(ModelDirect, self).__init__()
        self.testname = "where_operator_test_op_src_from_host"
        self.operator = operator
        self.treshold = 0.3

    def forward(self, x, y, z):
        condition = x > self.treshold
        return self.operator(condition, y, z)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "where_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.treshold = 0.3

        self.c1 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.c2 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=sum(shape),
        )
        self.c3 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=sum(shape) + 3,
        )
        self.register_buffer("constant1", self.c1)
        self.register_buffer("constant2", self.c2)
        self.register_buffer("constant3", self.c3)

    def forward(self, x, y, z):
        condition = self.c1 > self.treshold
        where_result = self.operator(condition, self.c2, self.c3)
        return where_result + x + y + z  # consuming the inputs to match the number of operands


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        warm_reset: bool = False,
    ):
        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        number_of_operands: int = 3
        value_range = ValueRanges.SMALL_POSITIVE

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(
                operator=operator,
                shape=test_vector.input_shape,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator=operator)
        )

        value_range = ValueRanges.SMALL

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])

        logger.trace(f"***input_shapes: {input_shapes}")

        # Using AllCloseValueChecker in all cases except for integer data formats
        if test_vector.dev_data_format in TestCollectionTorch.int.dev_data_formats:
            value_checker = ValueCheckerUtils.automatic()
        else:
            value_checker = ValueCheckerUtils.all_close()

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


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None


class TestCollectionData:

    __test__ = False

    operators = ["where"]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test all shapes and input sources collection:
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.operators,
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
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

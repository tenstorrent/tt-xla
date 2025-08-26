# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math

from typing import List
from loguru import logger

import random
import torch

from ...utils import (
    FailingReasons,
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
    def __init__(self, operator, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Reduce_pytorch_operator_test_op_src_from_another_op"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the reduce operator
        add1 = torch.add(x, x)
        output = self.operator(add1, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Reduce_pytorch_operator_test_op_src_from_host"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        output = self.operator(x, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Reduce_pytorch_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.kwargs = kwargs

        self.c1 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
        )
        self.register_buffer("constant", self.c1)

    def forward(self, x):
        v1 = self.operator(self.c1, **self.kwargs)
        v2 = self.operator(x, **self.kwargs)
        # add consume inputs
        add = torch.add(v1, v2)
        return add


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
        """Common verification function for all tests"""

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        value_range = ValueRanges.SMALL

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])
        logger.trace(f"***input_shapes: {input_shapes}")

        # We use AllCloseValueChecker in all cases except for integer data formats:
        value_checker = ValueCheckerUtils.all_close(atol=1e-2, rtol=1e-8)
        if test_vector.dev_data_format in TestCollectionTorch.int.dev_data_formats:
            value_checker = ValueCheckerUtils.automatic()

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

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        rng = random.Random(math.prod(test_vector.input_shape))
        dim = rng.choice(range(0, len(test_vector.input_shape)))
        for ch in [True, False]:
            yield {"dim": dim, "keepdim": ch}


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "sum",  # 00
            "mean",  # 01
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=[
            item
            for item in TestCollectionTorch.all.dev_data_formats
            if item not in TestCollectionTorch.single.dev_data_formats
        ],
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionTorch.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )

    sum = TestCollection(
        operators=[
            "sum",  # 00
        ],
    )

    mean = TestCollection(
        operators=[
            "mean",  # 01
        ],
        dev_data_formats=[
            item
            for item in TestCollectionTorch.float.dev_data_formats
            if item not in TestCollectionTorch.single.dev_data_formats
        ],
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test plan:
        # torch.sum(input), torch.mean(input)
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # torch.sum(input, kwargs), torch.mean(input, kwargs)
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Data format - sum operator
        TestCollection(
            operators=TestCollectionData.sum.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=TestCollectionData.all.dev_data_formats,
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Data format - mean operator
        TestCollection(
            operators=TestCollectionData.mean.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=TestCollectionData.mean.dev_data_formats,
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.all.operators),
        # Skip all tests with input shapes with 2 dimensions
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) == 2,
            skip_reason=FailingReasons.UNSUPPORTED_DIMENSION,
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
            dev_data_formats=[torch.float16],
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import math

from torch import nn

from typing import List
from loguru import logger

from ...utils import (
    FailingReasons,
    InputSource,
    PytorchUtils,
    TensorShape,
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

from ..eltwise_unary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


class ModelFromAnotherOpMax(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Max_operator_test_op_src_from_another_op"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        xx = torch.add(x, x)
        return self.operator(xx, **self.kwargs)[0]


class ModelDirectMax(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Max_operator_test_op_src_from_host"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        return self.operator(x, **self.kwargs)[0]


class ModelConstEvalPassMax(nn.Module):
    def __init__(self, operator, shape: TensorShape, kwargs, dtype, value_range: ValueRanges):
        super().__init__()
        self.testname = "Max_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.kwargs = kwargs
        self.c = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
        )
        self.register_buffer("constant", self.c)

    def forward(self, x):
        cc = self.operator(self.c, **self.kwargs)[0]
        xx = self.operator(x, **self.kwargs)[0]
        return torch.add(xx, cc)


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    MODEL_TYPES_MAX_SPECIFIC = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOpMax,
        InputSource.FROM_HOST: ModelDirectMax,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPassMax,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        warm_reset: bool = False,
    ):

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        value_range = ValueRanges.SMALL

        if not kwargs:
            # if kwargs is empty, max operator returns a tensor
            model_type = cls.MODEL_TYPES[test_vector.input_source]
        else:
            # if kwargs is not empty, max operator returns a tuple of tensors so we need a different model
            model_type = cls.MODEL_TYPES_MAX_SPECIFIC[test_vector.input_source]

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

    __test__ = False

    test_plan: TestPlan = None

    operator = ["max"]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):

        rng = random.Random(math.prod(test_vector.input_shape))

        dim = rng.choice(range(0, len(test_vector.input_shape)))

        for ch in [True, False]:
            yield {"dim": dim, "keepdim": ch}


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # torch.max(input)
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # torch.max(input, dim=..., keepdim=...)
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operator),
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            criteria=lambda test_vector: test_vector.kwargs is None,
            failing_reason=FailingReasons.COMPILATION_FAILED,
            # skip_reason="This test is expected to fail because the max operator is not supported for 'torch.max(input)' way of usage in forge (when skip_forge_verification is True all tests passed).",
            skip_reason=FailingReasons.COMPILATION_FAILED,
        ),
        TestCollection(
            operators=TestParamsData.operator,
            criteria=lambda test_vector: test_vector.get_id()
            == "max-FROM_HOST-{'dim': 1, 'keepdim': False}-(1, 2, 3, 4)-torch.float16-HiFi4",
            skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

from typing import List
from loguru import logger

from ...utils import (
    InputSource,
    PytorchUtils,
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

        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(
                operator, test_vector.input_shape, kwargs, dtype=test_vector.dev_data_format, value_range=value_range
            )
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])
        logger.trace(f"***input_shapes: {input_shapes}")

        # We use AllCloseValueChecker in all cases except for integer data formats(softmax doesn't support integer data formats):
        value_checker = ValueCheckerUtils.all_close(rtol=1e-2, atol=1e-2)

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

    operators = ["softmax"]

    @classmethod
    def generate_kwargs_pos(cls, test_vector: TestVector):
        for dim in range(len(test_vector.input_shape)):
            yield {"dim": dim}

    @classmethod
    def generate_kwargs_neg(cls, test_vector: TestVector):
        for dim in range(-1, -len(test_vector.input_shape) - 1, -1):
            yield {"dim": dim}

    @classmethod
    def generate_kwargs_all_but_last_dim_pos(cls, test_vector: TestVector):
        for dim in range(len(test_vector.input_shape) - 1):
            yield {"dim": dim}

    @classmethod
    def generate_kwargs_all_but_last_dim_neg(cls, test_vector: TestVector):
        for dim in range(-2, -len(test_vector.input_shape) - 1, -1):
            yield {"dim": dim}


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # TODO - Uncomment the following collections when the issue with the last dim value is resolved:
        # # Test all shapes and input sources collection:
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_sources=TestCollectionCommon.all.input_sources,
        #     input_shapes=TestCollectionCommon.all.input_shapes,
        #     kwargs=lambda test_vector: TestParamsData.generate_kwargs_pos(test_vector),
        # ),
        # # Test some shapes and input sources with negative dim values:
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_sources=TestCollectionCommon.all.input_sources,
        #     input_shapes=TestCollectionCommon.quick.input_shapes,
        #     kwargs=lambda test_vector: TestParamsData.generate_kwargs_neg(test_vector),
        # ),
        # Test all shapes and all input sources with last dim value(positive):
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: [{"dim": len(test_vector.input_shape) - 1}],
        ),
        # Test some shapes and all input sources with last dim value(negative):
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.quick.input_shapes,
            kwargs=lambda test_vector: [{"dim": -1}],
        ),
        # Test one shape and all input sources with all but last dim value(positive):
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_all_but_last_dim_pos(test_vector),
        ),
        # Test one shape and all input sources with all but last dim value(negative):
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_all_but_last_dim_neg(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": -1}],
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": -1}],
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operators),
        # # All dim values are not supported except for the last one:
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     criteria=lambda test_vector: test_vector.kwargs["dim"] != len(test_vector.input_shape) - 1
        #     and test_vector.kwargs["dim"] != -1,
        #     skip_reason=FailingReasons.UNSUPPORTED_AXIS,
        #     failing_reason=FailingReasons.UNSUPPORTED_AXIS,
        #     subcollections=[
        #         # One test case as check-flag to indicate that the dim value is not supported:
        #         TestCollection(
        #             operators=TestParamsData.operators,
        #             input_sources=TestCollectionCommon.single.input_sources,
        #             input_shapes=TestCollectionCommon.single.input_shapes,
        #             kwargs=[{"dim": 0}],
        #             # skip_reason=None,  # No need to explicit 'clear' the skip_reason
        #             failing_reason=FailingReasons.UNSUPPORTED_AXIS,
        #         ),
        #     ],
        # ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]

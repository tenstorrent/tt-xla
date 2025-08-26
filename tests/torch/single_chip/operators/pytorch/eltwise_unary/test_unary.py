# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
# 2. Operand source(s):
#    (+)  2.1 From another op
#           - Operator -> input
#    (+)  2.2 From DRAM queue - Removed from test plan
#           - Operator is first node in network
#           - Input_queue flag = false
#    (+)  2.3 Const Inputs (const eval pass)
#           - Operator where all inputs are constants.
#    (+)  2.4 From host
#           - Input tensor as input of network
#           - Operator is first node in network
#           - Input_queue flag = true
# 3. Tensor ranks:
#    (+)  3.1 Full tensor (i.e. full expected shape)
#           - 3-4 by default P1 (high prioriy)
#           - 2, 5, ++ include P2 (lower prioriy)
#    (+)  3.2 Tensor reduce on one or more dims to 1
#           - Vector
#           - Only one dim is not equal to 1
#    (-)  3.3 Scalar P2
#           - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
#    (+)  4.1 Divisible by 32
#    (+)  4.2 Prime numbers
#    (+)  4.3 Very large (thousands, 10s of thousands)
#           - 100x100, 100x1000
#           - maybe nightly only
#    (+)  4.4 Extreme ratios between height/width
#    (/)  4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
#    (/)  5.1 Output DF
#    (/)  5.2 Intermediate DF
#    (/)  5.3 Accumulation DF
#    (+)  5.4 Operand DFs
#           - Fix HiFi4 for math fidelity value
#    (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#           - Fix fp16b (default) for data format value
#    (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
#    (/) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
#    (/) Few representative values
#    (/) Reuse inputs for selected operators

import torch

from typing import List
from loguru import logger

from ...utils import (
    FailingReasons,
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

from .models import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


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

        dev_data_format = test_vector.dev_data_format
        if dev_data_format is None and test_vector.operator in TestCollectionData.bitwise.operators:
            # For bitwise operators, use int data format
            dev_data_format = TestCollectionData.single_int.dev_data_formats[0]

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(operator, test_vector.input_shape, kwargs, dtype=dev_data_format, value_range=value_range)
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

        # Using AllCloseValueChecker in all cases except for integer data formats
        if dev_data_format in TestCollectionTorch.int.dev_data_formats:
            value_checker = ValueCheckerUtils.automatic()
        else:
            value_checker = ValueCheckerUtils.all_close(rtol=1e-2, atol=1e-2)

        verify_config = VerifyConfig(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            dev_data_format=dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            value_range=value_range,
            warm_reset=warm_reset,
            value_checker=value_checker,
        )
        VerifyUtils.verify(verify_config)


class TestParamsData:

    __test__ = False

    test_plan_implemented: TestPlan = None
    test_plan_implemented_float: TestPlan = None
    test_plan_not_implemented: TestPlan = None

    no_kwargs = [
        None,
    ]

    kwargs_clamp = [
        {"min": 0.0, "max": 0.5},
        {"min": 0.5, "max": 0.0},
        {"min": 0.2},
        {"max": 0.2},
    ]

    kwargs_pow = [
        {"exponent": 0.5},
        {"exponent": 2.0},
        {"exponent": 10.0},
    ]

    kwargs_gelu = [
        {"approximate": "tanh"},
        {},
    ]

    kwargs_leaky_relu = [
        {"negative_slope": 0.01, "inplace": True},
        {"negative_slope": 0.1, "inplace": False},
        {},
    ]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        if test_vector.operator in ("clamp",):
            return cls.kwargs_clamp
        if test_vector.operator in ("pow",):
            return cls.kwargs_pow
        if test_vector.operator in ("gelu",):
            return cls.kwargs_gelu
        if test_vector.operator in ("leaky_relu",):
            return cls.kwargs_leaky_relu
        if test_vector.operator in ("cumsum",):
            return [{"dim": d} for d in range(0, len(test_vector.input_shape))]
        return cls.no_kwargs


class TestCollectionData:

    __test__ = False

    implemented = TestCollection(
        operators=[
            "relu",
            "sqrt",
            "reciprocal",
            "sigmoid",
            "abs",
            "cos",
            "exp",
            "neg",
            "rsqrt",
            "sin",
            "square",
            "pow",
            "clamp",
            "log",
            "log1p",
            "cumsum",
            "isnan",
            "tanh",
        ],
    )
    implemented_float = TestCollection(
        operators=[
            "gelu",
            "leaky_relu",
        ],
    )
    not_implemented = TestCollection(
        operators=[
            "acos",
            "acosh",
            "angle",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "bitwise_not",
            "ceil",
            "conj_physical",
            "cosh",
            "deg2rad",
            "digamma",
            "erf",
            "erfc",
            "erfinv",
            "exp2",
            "expm1",
            "floor",
            "frac",
            "lgamma",
            "log10",
            "log2",
            "logit",
            "i0",
            "nan_to_num",
            "positive",
            "rad2deg",
            "round",
            "sign",
            "sgn",
            "signbit",
            "sinc",
            "sinh",
            "tan",
            "trunc",
        ],
    )

    bitwise = TestCollection(
        operators=[
            "bitwise_not",
        ],
    )

    all_int = TestCollection(
        dev_data_formats=TestCollectionTorch.int.dev_data_formats,
    )

    single_int = TestCollection(
        dev_data_formats=TestCollectionTorch.int.dev_data_formats[0:1],
    )
    # torch.float16 is not supported well - python crashes
    common_to_skip = TestCollection(
        dev_data_formats=[torch.float16],
        failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


TestParamsData.test_plan_implemented = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test Special cases collection for "pow" operator:
        TestCollection(
            operators=["pow"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.specific.input_shapes,
            kwargs=[
                {"exponent": 0.0},
                {"exponent": 0.5},
                {"exponent": 2.0},
                {"exponent": 10.0},
                {"exponent": -2.0},
                {"exponent": -1.26},
                {"exponent": 1.52},
            ],
            dev_data_formats=TestCollectionTorch.all.dev_data_formats,
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(
            operators=TestCollectionData.implemented.operators,
        ),
        TestCollectionData.common_to_skip,
    ],
)


TestParamsData.test_plan_implemented_float = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test gelu, leaky_relu operators collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test gelu, leaky_relu data formats collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test gelu, leaky_relu math fidelities collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(
            operators=TestCollectionData.implemented_float.operators,
        ),
        TestCollectionData.common_to_skip,
    ],
)


TestParamsData.test_plan_not_implemented = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        )
    ],
    failing_rules=[
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            failing_reason=FailingReasons.NOT_IMPLEMENTED_ATEN,
        ),
        TestCollection(
            operators=[
                "acos",
                "asin",
                "ceil",
                "cosh",
                "log10",
                "log2",
                "round",
                "sign",
                "sinh",
                "tan",
                "trunc",
            ],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            failing_reason=FailingReasons.UNSUPPORTED_OP_TYPES,
        ),
        TestCollection(
            operators=TestCollectionData.bitwise.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            failing_reason=FailingReasons.UNSUPPORTED_OP_TYPES,
        ),
        TestCollectionData.common_to_skip,
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan_implemented,
        TestParamsData.test_plan_implemented_float,
        TestParamsData.test_plan_not_implemented,
    ]

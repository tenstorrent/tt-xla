# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import math

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
    def __init__(self, operator, opname, kwargs):
        super().__init__()
        self.testname = f"{opname}_operator_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.batchnorm2d = self.operator(**kwargs)

    def forward(self, x):
        xx = torch.add(x, x)
        return self.batchnorm2d(xx)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, opname, kwargs):
        super().__init__()
        self.testname = f"{opname}_operator_test_op_src_direct"
        self.operator = operator
        self.opname = opname
        self.batchnorm2d = self.operator(**kwargs)

    def forward(self, x):
        return self.batchnorm2d(x)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, opname, input_shape, kwargs, value_range):
        super().__init__()
        self.testname = f"{opname}_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.batchnorm2d = self.operator(**kwargs)
        self.dtype = kwargs.get("dtype")
        self.const = TensorUtils.create_torch_constant(
            input_shape=input_shape,
            dev_data_format=self.dtype,
            value_range=value_range,
        )
        self.register_buffer("constant", self.const)

    def forward(self, x):
        cc = self.batchnorm2d(self.const)
        xx = self.batchnorm2d(x)
        return torch.add(cc, xx)


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
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                input_shape=test_vector.input_shape,
                kwargs=kwargs,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                kwargs=kwargs,
            )

        dtype = kwargs.get("dtype")

        input_shapes = tuple([test_vector.input_shape])
        logger.trace(f"***input_shapes: {input_shapes}")

        # Using AllCloseValueChecker
        value_checker = ValueCheckerUtils.all_close(rtol=1e-2, atol=1e-2)

        verify_config = VerifyConfig(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            model_dtype=dtype if dtype is torch.bfloat16 else None,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_range=value_range,
            value_checker=value_checker,
        )
        VerifyUtils.verify(verify_config)


class TestParamsData:

    __test__ = False

    test_plan_1d: TestPlan = None
    test_plan_2d: TestPlan = None
    test_plan_2d: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        kwargs = {
            "num_features": test_vector.input_shape[1],
            "dtype": test_vector.dev_data_format,
        }
        yield kwargs

    @classmethod
    def generate_kwargs_eps_momentum(cls, test_vector: TestVector):
        for eps in [1e-1, 1e-3, 1e-5, 1e-12]:
            for momentum in [0.0, 0.1, 0.9, 1.0, None]:
                yield {
                    "num_features": test_vector.input_shape[1],
                    "eps": eps,
                    "momentum": momentum,
                }


class TestCollectionData:
    __test__ = False

    operator_1d = ["batch_norm_1d"]
    operator_2d = ["batch_norm_2d"]
    operator_3d = ["batch_norm_3d"]

    input_shapes_1d = [s for s in TestCollectionCommon.all.input_shapes if len(s) in [2, 3]]
    input_shapes_2d = [s for s in TestCollectionCommon.all.input_shapes if len(s) == 4]

    @staticmethod
    def _input_shapes_3d(shapes):
        ret = []
        for s in shapes:
            rnd = random.Random(math.prod(s))
            ret.append(s[:2] + (rnd.randint(1, 5),) + s[2:])
        return ret

    input_shapes_3d = _input_shapes_3d(input_shapes_2d)

    single_input_shape_1d = [(32, 64, 64)]
    single_input_shape_2d = [(10, 32, 64, 64)]
    single_input_shape_3d = [(10, 32, 5, 64, 64)]


TestParamsData.test_plan_1d = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.input_shapes_1d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_1d,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Math fidelities collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_1d,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test eps and momentum collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.single_input_shape_1d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_eps_momentum(test_vector),
        ),
        # Test affine = False collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_1d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "affine": False,
                    }
                ]
            ),
        ),
        # Test track_running_stats = False collection
        TestCollection(
            operators=TestCollectionData.operator_1d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_1d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "track_running_stats": False,
                    }
                ]
            ),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operator_1d),
    ],
)


TestParamsData.test_plan_2d = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.input_shapes_2d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_2d,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Math fidelities collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_2d,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test eps and momentum collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.single_input_shape_2d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_eps_momentum(test_vector),
        ),
        # Test affine = False collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_2d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "affine": False,
                    }
                ]
            ),
        ),
        # Test track_running_stats = False collection
        TestCollection(
            operators=TestCollectionData.operator_2d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_2d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "track_running_stats": False,
                    }
                ]
            ),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operator_2d),
    ],
)


TestParamsData.test_plan_3d = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.input_shapes_3d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_3d,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Math fidelities collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_3d,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test eps and momentum collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.single_input_shape_3d,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_eps_momentum(test_vector),
        ),
        # Test affine = False collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_3d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "affine": False,
                    }
                ]
            ),
        ),
        # Test track_running_stats = False collection
        TestCollection(
            operators=TestCollectionData.operator_3d,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionData.single_input_shape_3d,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "track_running_stats": False,
                    }
                ]
            ),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operator_3d),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan_1d,
        TestParamsData.test_plan_2d,
        TestParamsData.test_plan_3d,
    ]

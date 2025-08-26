# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from typing import List
from loguru import logger

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

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Concatenate_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, *x: torch.Tensor):
        # we use Add operator to create one operands which is input for the concatenate operator
        add_tuple = ()
        for i in range(len(x)):
            add_tuple += (torch.add(x[i], x[i]),)
        output = self.operator(add_tuple, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Concatenate_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, *x: torch.Tensor):
        output = self.operator(x, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, dtype=None, value_range=None):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Concatenate_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        g = torch.Generator().manual_seed(17)
        const_shape = shape[0]
        self.constants = tuple()
        for i in range(len(shape)):
            random_seed = torch.randint(low=math.prod(const_shape), high=2**32 - 1, size=(1,), generator=g).item()
            g.manual_seed(random_seed)
            self.constants += (
                TensorUtils.create_torch_constant(
                    input_shape=const_shape,
                    dev_data_format=dtype,
                    value_range=value_range,
                    random_seed=random_seed,
                ),
            )
            self.register_buffer(f"constant{i}", self.constants[i])

    def forward(self, *x: torch.Tensor):
        v1 = self.operator(self.constants, **self.kwargs)
        v2 = torch.add(x[0], x[1])
        for i in range(2, len(x)):
            v3 = torch.add(v2, x[i])
            v2 = v3
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

        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(
                operator=operator, opname=test_vector.operator, shape=test_vector.input_shape, kwargs=kwargs
            )
        )

        dim = test_vector.kwargs["dim"]
        num_operands = len(test_vector.input_shape)
        match test_vector.input_source:
            case InputSource.CONST_EVAL_PASS:
                input_shape = list(test_vector.input_shape[0])
                input_shape[dim] = input_shape[dim] * num_operands
                input_shapes = tuple([input_shape for _ in range(num_operands)])
            case _:
                input_shapes = test_vector.input_shape

        logger.trace(f"***input_shapes: {input_shapes}")

        # Using AllCloseValueChecker in all cases except for integer data formats:
        value_checker = ValueCheckerUtils.all_close(atol=1e-2)
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

    operators = ["concatenate"]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        for i in range(len(test_vector.input_shape[0])):
            yield {"dim": i}

    @classmethod
    def generate_input_shapes(cls):
        shapes = TestCollectionCommon.all.input_shapes
        number_of_operands = [
            2,
            3,
            7,
            # 15, # consume too much memory
        ]
        for shape in shapes:
            for repeat in number_of_operands:
                yield (shape,) * repeat


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
            input_shapes=TestParamsData.generate_input_shapes(),
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=[(TestCollectionCommon.single.input_shapes[0],) * 2],
            kwargs=[{"dim": 0}],
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
            input_shapes=[(TestCollectionCommon.single.input_shapes[0],) * 2],
            kwargs=[{"dim": 0}],
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operators),
        # Unsupported ttnn::DataType... Fatal Python error: Aborted
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=[(TestCollectionCommon.single.input_shapes[0],) * 2],
            kwargs=[{"dim": 0}],
            dev_data_formats=[torch.float16],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
            skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # # Unsupported special cases:
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_sources=[
        #         InputSource.FROM_ANOTHER_OP,
        #         InputSource.FROM_HOST,
        #     ],
        #     input_shapes=[
        #         ((1, 10000), (1, 10000), (1, 10000)),
        #         ((1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000)),
        #     ],
        #     kwargs=[{"dim": 1}],
        #     failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        # ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

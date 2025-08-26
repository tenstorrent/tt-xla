# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of embedding operators
#
# In this test we test pytorch embedding operator


from functools import reduce
import math
import random

from typing import List, Type

import torch
from loguru import logger

from ...utils import (
    InputSource,
    PytorchUtils,
    TensorUtils,
    TestCollection,
    TestCollectionCommon,
    TestDevice,
    TestPlan,
    TestVector,
    ValueCheckerUtils,
    ValueRange,
    VerifyConfig,
    VerifyUtils,
)
from ..ids import TestIdsDataLoader


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
            "dtype": kwargs["weight_dtype"],
        }

        self.embedding = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the embedding operator
        add = torch.add(x, x)
        output = self.embedding(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
            "dtype": kwargs["weight_dtype"],
        }

        self.embedding = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.embedding(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
            "dtype": kwargs["weight_dtype"],
        }

        self.const = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=kwargs.get("input_dtype"),
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.register_buffer("constant", self.const)

        self.embedding = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        v1 = self.embedding(self.constant)
        # v2 = torch.add(x, x)
        v2 = self.embedding(x)
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
        number_of_operands: int = 1,
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)

        min = 0
        max = test_vector.kwargs["num_embeddings"] - 1
        if test_vector.input_source == InputSource.FROM_ANOTHER_OP:
            max = int(max / 2)
        value_range = ValueRange(min, max)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
            )

        dtype = kwargs.get("weight_dtype")

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        # We test only int data type for inputs but we AllCloseValueChecker instead of AutomaticValueChecker because outputs of embedding operator are always float
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
            value_checker=value_checker,
            value_range=value_range,
        )
        VerifyUtils.verify(verify_config)


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    # rng = random.Random(31)
    INPUT_SHAPE_THRESHOLD = 100000000
    MAX_EMBEDDING_DIM = 10000

    embedding_dims = [1000, 3200, MAX_EMBEDDING_DIM]

    @classmethod
    def generate_kwargs(
        cls,
        test_vector: TestVector,
        weight_dtype: Type[torch.dtype] = None,
        num_embeddings_min: int = 2,
        num_embeddings_max: int = 32000,
    ):

        rng = random.Random(math.prod(test_vector.input_shape) + 1)
        num_embeddings = [rng.randint(num_embeddings_min, num_embeddings_max)]

        kwarg_list = []
        for num_embeddings in num_embeddings:
            for embedding_dim in cls.embedding_dims:
                kwarg_list.append(
                    {
                        "num_embeddings": num_embeddings,
                        "embedding_dim": embedding_dim,
                        "weight_dtype": weight_dtype,
                        "input_dtype": test_vector.dev_data_format,
                    }
                )
        return kwarg_list


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "embedding",
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=[
            input_shape
            for input_shape in TestCollectionCommon.all.input_shapes
            if reduce(lambda x, y: x * y, input_shape) * TestParamsData.MAX_EMBEDDING_DIM
            < TestParamsData.INPUT_SHAPE_THRESHOLD
        ],
        dev_data_formats=[torch.int32, torch.int64],
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=[
            torch.int32,
        ],
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test plan:
        # 2. Operand source(s):
        # 3. Operand shapes type(s):
        # 4. Operand / output size of dimensions
        # Case of weight dtype torch.float32
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, weight_dtype=torch.float32),
        ),
        # Case of num_embeddings range
        # between 32000 and 500000
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(
                test_vector,
                num_embeddings_min=32000,
                num_embeddings_max=500000,
            ),
        ),
        # Case of all sources and shapes from the test plan
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            dev_data_formats=TestCollectionData.all.dev_data_formats,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, weight_dtype=torch.bfloat16),
        ),
        # Case of all math fidelities
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, weight_dtype=torch.bfloat16),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=["embedding"]),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

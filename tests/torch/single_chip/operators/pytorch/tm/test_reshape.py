# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# TODO: Add test plan header here

import math
import random

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


def prime_factors(n):
    """Return the prime factors of a given number."""
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i

    if n > 2:
        factors.append(n)

    return factors


def generate_random_shape(source_shape, max_dims=4):
    """
    Generates a random shape with the same volume as the source shape.
    Args:
        source_shape (tuple): The original shape of the tensor.
        max_dims (int, optional): The maximum number of dimensions for the generated shape. Defaults to 4.
    Returns:
        tuple: A randomly generated shape with the same volume as the source shape.
    Raises:
        ValueError: If the volume or number of dimensions is not a positive integer.
        ValueError: If the generated shape does not match the desired volume.
    Notes:
        - The function ensures that the product of the dimensions of the generated shape equals the product of the dimensions of the source shape.
        - If the generated shape is the same as the source shape, an additional dimension with size 1 is appended to the shape.
    """

    rng = random.Random(math.prod(source_shape) + 1)

    vol = math.prod(source_shape)
    num_dims = rng.randint(1, max_dims)

    if vol <= 0 or num_dims <= 0:
        raise ValueError("Volume and number of dimensions must be positive integers.")

    # Step 1: Get the prime factors of vol
    factors = prime_factors(vol)

    # Step 2: Start with dimensions initialized to 1
    shape = [1] * num_dims

    # Step 3: Randomly assign factors to dimensions
    for factor in factors:
        random_dim = rng.randint(0, num_dims - 1)
        shape[random_dim] *= factor

    # Step 4: Ensure the final shape has the correct volume
    if math.prod(shape) != vol:
        raise ValueError("Generated shape does not match the desired volume.")

    return tuple(shape)


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

        value_range = ValueRanges.LARGE
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                kwargs=kwargs,
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

    # fmt: off
    specific_reshapes = {
        # Flatten Reshapes:
        (8, 8, 8):        [(512,)],
        (1, 2, 2, 2):     [(4, 2), (8,), ],
        # Dynamic Shape Reshapes:
        (1, 49, 2304):    [(1, -1, 3, 24, 32),
                           (1, 49, 6, -1),
                           (-1, ),
                           (1, -1)],
        (3, 4, 5):        [(3, -1), (-1, 15)],
        (2, 2, 2, 2):     [(4, -1), (8, -1)],
        # Collapse Rank Reshapes:
        (2, 3, 4, 5):     [(6, 4, 5), (4, 6, 5)],
        (2, 2, 3, 4):     [(4, 3, 4), (2, 6, 4)],
        (3, 3, 3, 3):     [(9, 3, 3), (3, 9, 3)],
        (1, 2, 3, 4, 5):  [(2, 3, 4, 5), (6, 4, 5)],
        (2, 2, 2, 2, 2):  [(4, 2, 2, 2), (8, 2, 2)],
        (2, 3, 4, 8):     [(6, 4, 8), (24, 8)],
        # Expand Rank Reshapes:
        (6, 4, 5):        [(2, 3, 4, 5),
                           (1, 6, 4, 5),
                           (2, 3, 2, 2, 5)],
        (12, 8, 10):      [(2, 6, 8, 10),
                           (2, 2, 3, 8, 10),
                           (1, 3, 4, 8, 10)],
        # Channel Manipulation Reshapes:
        (1, 32, 2560):    [(1, 32, 20, 128),
                           (1, 32, 40, 64),
                           (1, 32, 10, 256),
                           (1, 32, 5, 512),
                           (1, 32, 80, 32)],
        (1, 64, 1280):    [(1, 64, 10, 128),
                           (1, 64, 20, 64),
                           (1, 64, 5, 256),
                           (1, 64, 40, 32),
                           (1, 64, 80, 16)],
        # Edge Case Reshapes:
        (0,):             [(0, 0, 1),
                           (0, 1, 0),
                           (1, 0, 0),
                           (0, 0, 10)],
    }
    # fmt: on

    @classmethod
    def generate_random_kwargs(cls, test_vector: TestVector):
        """
        This method generates a random target shape based on the input shape of the
        test vector and a seed value derived from the sum of the elements in the
        input shape. The generated target shape is then returned as a dictionary
        within a list.
        Args:
            test_vector (TestVector): The test vector containing the input shape.
        Returns:
            list: A list containing a dictionary with the generated target shape.
        """

        input_shape = test_vector.input_shape
        seed = 0
        for i in range(len(input_shape)):
            seed += input_shape[i]
        target_shape = generate_random_shape(test_vector.input_shape)
        return [
            {"shape": target_shape},
        ]

    @classmethod
    def generate_specific_kwargs(cls, test_vector: TestVector):
        """
        Generate specific keyword arguments for reshaping tensors based on the given test vector.
        Args:
            cls: The class that contains specific reshapes mapping.
            test_vector (TestVector): An instance of TestVector containing the input shape.
        Returns:
            list: A list of dictionaries, each containing a target shape for reshaping.
        """

        input_shape = test_vector.input_shape
        target_shapes = []
        for item in cls.specific_reshapes[input_shape]:
            target_shapes.append({"shape": item})
        return target_shapes


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test specific classes of reshape operations collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestParamsData.specific_reshapes.keys(),
            kwargs=lambda test_vector: TestParamsData.generate_specific_kwargs(test_vector),
        ),
    ],
    failing_rules=[
        # TestCollection(
        #     input_shapes=[(1, 10000), (7, 10, 1000, 100)],
        #     failing_reason=FailingReasons.INFERENCE_FAILED,
        # ),
        # TestCollection(
        #     input_shapes=[(0,)],
        #     failing_reason=FailingReasons.UNSUPPORTED_DIMENSION,
        # ),
        *TestIdsDataLoader.build_failing_rules(operators=["reshape"]),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]

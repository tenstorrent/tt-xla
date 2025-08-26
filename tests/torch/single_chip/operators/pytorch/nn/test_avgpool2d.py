# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Tests for testing of linear operators
#
# In this test we test pytorch avgpool2d operator

from dataclasses import dataclass
import math
import random

from typing import List, Optional, Tuple, Union
from loguru import logger

import torch

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

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "AvgPool2d_pytorch_operator_test_src_from_another_op"
        self.operator = operator
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)
        # self.ct1 = nn.AvgPool2d(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the AvgPool2d operator
        add = torch.add(x, x)
        output = self.ct1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "AvgPool2d_pytorch_operator_test_src_from_host"
        self.operator = operator
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)
        # self.ct1 = nn.AvgPool2d(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.ct1(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "AvgPool2d_pytorch_operator_test_src_const_eval_pass"
        self.operator = operator
        self.shape = shape
        self.kwargs = kwargs

        self.constant = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.register_buffer("constant1", self.constant)

        self.ct1 = self.operator(**self.kwargs)
        # self.ct1 = nn.AvgPool2d(**self.kwargs)

    def forward(self, x: torch.Tensor):
        v1 = self.ct1(self.constant)
        # v2 = torch.add(x, x)
        v2 = self.ct1(x)
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

        value_range = ValueRanges.SMALL
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
                shape=test_vector.input_shape,
                kwargs=kwargs,
            )

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        value_checker = ValueCheckerUtils.automatic(pcc=0.99, rtol=1e-2, atol=1e-2)

        verify_config = VerifyConfig(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_checker=value_checker,
            value_range=value_range,
        )
        VerifyUtils.verify(verify_config)


@dataclass
class AvgPool2DParams:
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    ceil_mode: bool = False
    count_include_pad: bool = True
    divisor_override: Optional[int] = None

    max_kernel_height = 100
    max_kernel_width = 100

    @classmethod
    def to_kwargs(self):
        return self.__dict__

    def to_kwargs_list(self):
        return [self.__dict__]

    @classmethod
    def get_kernel_param(cls, rng, H_in, W_in, k_min_h=1, k_min_w=1, odd: bool = False):
        k_min_h = min(k_min_h, H_in)
        k_min_w = min(k_min_w, W_in)

        # Adjust minimum values to be odd if required
        if odd:
            if k_min_h % 2 == 0:
                k_min_h = max(k_min_h + 1, 1)
            if k_min_w % 2 == 0:
                k_min_w = max(k_min_w + 1, 1)

        # Initial limit based on input size
        k_h_limit = min(H_in // 3, cls.max_kernel_height)
        k_w_limit = min(W_in // 3, cls.max_kernel_width)

        # Ensure minimum size constraints
        k_h_limit = max(k_h_limit, k_min_h)
        k_w_limit = max(k_w_limit, k_min_w)

        # Generate kernel sizes within the computed limits
        k_h = rng.randint(k_min_h, k_h_limit)
        k_w = rng.randint(k_min_w, k_w_limit)

        # Ensure odd-sized kernels if required
        if odd:
            if k_h % 2 == 0:
                k_h = max(k_h - 1, k_min_h) if k_h - 1 >= k_min_h else k_h + 1
            if k_w % 2 == 0:
                k_w = max(k_w - 1, k_min_w) if k_w - 1 >= k_min_w else k_w + 1

        return k_h, k_w

    @classmethod
    def get_non_unit_stride_param(cls, rng, H_in, W_in, k_eff_h, k_eff_w):
        """
        Generates a non-unit stride parameter for a AvgPool2D operation.
        The stride is calculated based on the effective kernel size and the input dimensions.

        Parameters:
            rng (random.Random): Random number generator.
            H_in (int): Height of the input tensor.
            W_in (int): Width of the input tensor.
            k_eff_h (int): Effective kernel height.
            k_eff_w (int): Effective kernel width.

        Returns:
            tuple: A tuple representing the stride (height, width).
        """
        stride_h_max = H_in - k_eff_h + 1
        stride_h_max = max(stride_h_max // 5, 2)
        stride_w_max = W_in - k_eff_w + 1
        stride_w_max = max(stride_w_max // 5, 2)
        stride_h = rng.randint(2, stride_h_max) if H_in > 1 else 1
        stride_w = rng.randint(2, stride_w_max) if W_in > 1 else 1
        stride = (stride_h, stride_w)
        return stride

    @classmethod
    def get_padding_param(cls, rng, kernel_size: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
        if isinstance(kernel_size, int):
            max_pad = kernel_size // 2
            return rng.randint(0, max_pad)
        elif isinstance(kernel_size, tuple):
            return tuple(rng.randint(0, k // 2) for k in kernel_size)
        else:
            return 0  # No padding if kernel_size is not an int or tuple


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs_all(cls, test_vector: TestVector):
        kwarg_list = []
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides(test_vector))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_unit_strides(test_vector))
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_no_unit_strides(test_vector))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_no_unit_strides(test_vector))
        kwarg_list.extend(cls.generate_kwargs_same_output_as_input(test_vector))
        return kwarg_list

    @classmethod
    def generate_kwargs_single(cls, test_vector: TestVector):
        kwarg_list = []
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides(test_vector))
        return kwarg_list

    ## no zero padding
    @classmethod
    def generate_kwargs_no_zero_padding_unit_strides(cls, test_vector: TestVector, ceil_mode: bool = False):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        avgPool2DParams = AvgPool2DParams(
            kernel_size=AvgPool2DParams.get_kernel_param(rng, H_in, W_in),
            ceil_mode=ceil_mode,
        )
        # create kwargs
        return avgPool2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_no_zero_padding_no_unit_strides(cls, test_vector: TestVector, ceil_mode: bool = False):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # skip if input is 1x1, not possible to have non-unit stride tested
        if H_in == 1 and W_in == 1:
            return []
        # prepare params
        kernel_size = AvgPool2DParams.get_kernel_param(rng, H_in, W_in)
        stride = AvgPool2DParams.get_non_unit_stride_param(rng, H_in, W_in, kernel_size[0], kernel_size[1])
        avgPool2DParams = AvgPool2DParams(
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
        return avgPool2DParams.to_kwargs_list()

    ## with zero padding
    @classmethod
    def generate_kwargs_zero_padding_unit_strides(
        cls, test_vector: TestVector, ceil_mode: bool = False, count_include_pad: bool = True
    ):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        kernel_size = AvgPool2DParams.get_kernel_param(rng, H_in, W_in)
        padding = AvgPool2DParams.get_padding_param(rng, kernel_size)
        if isinstance(padding, int):
            if padding == 0:
                return []
        elif isinstance(padding, tuple):
            if all(p == 0 for p in padding):
                return []
        # prepare params
        avgPool2DParams = AvgPool2DParams(
            kernel_size=kernel_size,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        return avgPool2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_no_unit_strides(
        cls, test_vector: TestVector, ceil_mode: bool = False, count_include_pad: bool = True
    ):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # skip if input is 1x1, not possible to have non-unit stride tested
        if H_in == 1 and W_in == 1:
            return []
        # prepare params
        kernel_size = AvgPool2DParams.get_kernel_param(rng, H_in, W_in)
        stride = AvgPool2DParams.get_non_unit_stride_param(rng, H_in, W_in, kernel_size[0], kernel_size[1])
        padding = AvgPool2DParams.get_padding_param(rng, kernel_size)
        if isinstance(padding, int):
            if padding == 0:
                return []
        elif isinstance(padding, tuple):
            if all(p == 0 for p in padding):
                return []
        avgPool2DParams = AvgPool2DParams(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        return avgPool2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_same_output_as_input(
        cls, test_vector: TestVector, ceil_mode: bool = False, count_include_pad: bool = True
    ):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # H_out = H_in padding = floor(kernel_size / 2), stride = 1 - same for width
        # prepare params
        kernel_size = AvgPool2DParams.get_kernel_param(rng, H_in, W_in, k_min_h=2, k_min_w=2, odd=True)
        if kernel_size[0] == 1 or kernel_size[1] == 1:
            return []
        padding_h = ((kernel_size[0]) if (kernel_size[0]) > 0 else 1) // 2
        padding_w = ((kernel_size[1]) if (kernel_size[1]) > 0 else 1) // 2
        padding = (padding_h, padding_w)
        if padding_h == 0 and padding_w == 0:
            return []
        avgPool2DParams = AvgPool2DParams(
            kernel_size=kernel_size,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        return avgPool2DParams.to_kwargs_list()


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "avg_pool_2d",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        # only 4D input tensors are supported
        input_shapes=[input_shape for input_shape in TestCollectionCommon.all.input_shapes if len(input_shape) == 4],
        dev_data_formats=TestCollectionTorch.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=[
            (3, 11, 45, 17),
        ],
        dev_data_formats=TestCollectionTorch.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )

    float = TestCollection(
        dev_data_formats=TestCollectionTorch.float.dev_data_formats,
    )


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test plan:
        ## No padding, unit strides, ceil_mode=False
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
        ),
        ## No padding, unit strides, ceil_mode=True
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(
                test_vector, ceil_mode=True
            ),
        ),
        ## No padding, no unit strides, ceil_mode=False
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_no_unit_strides(test_vector),
        ),
        ## With zero padding, unit strides, ceil_mode=False, count_include_pad=True
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides(test_vector),
        ),
        ## With zero padding, unit strides, ceil_mode=False, count_include_pad=False
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides(
                test_vector, count_include_pad=False
            ),
        ),
        ## With zero padding, no unit strides, ceil_mode=False, count_include_pad=True
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_no_unit_strides(test_vector),
        ),
        ## With zero padding but to make same output as input, no unit strides, ceil_mode=False, count_include_pad=True
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_same_output_as_input(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionData.float.dev_data_formats  # int data formats not supported
                if item not in TestCollectionData.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.all.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

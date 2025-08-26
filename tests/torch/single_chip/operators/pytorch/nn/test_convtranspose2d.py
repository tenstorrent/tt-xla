# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import math
import os
import random

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
    TestPlanUtils,
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
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the ConvTranspose2d operator
        add = torch.add(x, x)
        output = self.ct1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.ct1(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.constant = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=kwargs.get("dtype"),
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.register_buffer("constant1", self.constant)

        self.ct1 = self.operator(**self.kwargs)

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

        dtype = kwargs.get("dtype")

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        value_checker = ValueCheckerUtils.automatic(pcc=0.99, rtol=1e-2, atol=1e-2)

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


@dataclass
class ConvTranspose2DParams:
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    groups: int = 1
    bias: bool = True
    dilation: int = 1
    dtype: torch.dtype = None

    max_kernel_height = 100
    max_kernel_width = 100

    @classmethod
    def to_kwargs(cls):
        return cls.__dict__

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
        Generates a non-unit stride parameter for a Conv2D operation.
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
    def get_groups_param(cls, in_channels, out_channels, mode="standard"):
        """
        Generates the `groups` parameter for a Conv2D operation.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mode (str): Convolution mode - "standard", "depthwise", or "grouped".

        Returns:
            int: The appropriate groups value.
        """
        if mode == "standard":
            return 1  # Standard convolution (all input channels connect to all output channels)
        elif mode == "depthwise":
            return in_channels if in_channels == out_channels else 1  # Depthwise only valid if channels match
        elif mode == "grouped":
            return math.gcd(in_channels, out_channels)  # Largest common divisor for efficient grouping
        else:
            raise ValueError("Invalid mode. Choose from 'standard', 'depthwise', or 'grouped'.")

    @classmethod
    def find_k_d(cls, k_eff):
        """
        Finds all possible pairs (k, d) that satisfy the equation:

            k + (k - 1) * (d - 1) = k_eff

        where:
        - `k` is a positive integer starting from 1.
        - `d` is a positive integer starting from 2.

        Parameters:
        -----------
        k_eff : int
            The effective kernel size (upper bound is capped at 100).

        Returns:
        --------
        list of tuple:
            A list of tuples `(k, d)`, where each tuple represents a valid pair
            that satisfies the equation.
        """
        results = []
        if k_eff > 100:
            k_eff = 100
        for k in range(1, k_eff + 1):  # Trying values from 1 to k_eff
            for d in range(2, k_eff + 1):
                if k + (k - 1) * (d - 1) == k_eff:
                    results.append((k, d))
        return results

    def is_output_shape_valid(self, H_in, W_in):
        k_h = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        k_w = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        p_h = self.padding if isinstance(self.padding, int) else self.padding[0]
        p_w = self.padding if isinstance(self.padding, int) else self.padding[1]
        s_h = self.stride if isinstance(self.stride, int) else self.stride[0]
        s_w = self.stride if isinstance(self.stride, int) else self.stride[1]
        d_h = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        d_w = self.dilation if isinstance(self.dilation, int) else self.dilation[1]
        op_h = self.output_padding if isinstance(self.output_padding, int) else self.output_padding[0]
        op_w = self.output_padding if isinstance(self.output_padding, int) else self.output_padding[1]

        H_out = (H_in - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
        W_out = (W_in - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1
        if H_out <= 0 or W_out <= 0:
            return False
        return True


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs_all(cls, test_vector: TestVector, bias: bool = True):
        kwarg_list = []
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_no_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_no_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides_dilation(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_unit_strides_dilation(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_no_unit_strides_dilation(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_groups_depthwise(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_groups_grouped(test_vector, bias))
        return kwarg_list

    @classmethod
    def generate_kwargs_no_zero_padding_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in),
            bias=bias,
            dtype=test_vector.dev_data_format,
        )
        # create kwargs
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in),
            padding=rng.randint(1, 4),
            bias=bias,
        )
        if conv2DParams.is_output_shape_valid(H_in, W_in) is False:
            return []
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_no_zero_padding_no_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # skip if input is 1x1, not possible to have non-unit stride tested
        if H_in == 1 and W_in == 1:
            return []
        # prepare params
        kernel_size = ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in)
        stride = ConvTranspose2DParams.get_non_unit_stride_param(rng, H_in, W_in, kernel_size[0], kernel_size[1])
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_no_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # skip if input is 1x1, not possible to have non-unit stride tested
        if H_in == 1 and W_in == 1:
            return []
        # prepare params
        kernel_size = ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in)
        stride = ConvTranspose2DParams.get_non_unit_stride_param(rng, H_in, W_in, kernel_size[0], kernel_size[1])
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=kernel_size,
            stride=stride,
            padding=rng.randint(1, 4),
            bias=bias,
        )
        if conv2DParams.is_output_shape_valid(H_in, W_in) is False:
            return []
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_no_zero_padding_unit_strides_dilation(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        # k_eff = k + (k − 1)(d − 1).
        k_eff_h = rng.randint(H_in // 2 if H_in // 2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in // 2 if W_in // 2 > 0 else 1, W_in)
        possible_k_d_h = ConvTranspose2DParams.find_k_d(k_eff_h)
        possible_k_d_w = ConvTranspose2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_unit_strides_dilation(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        # k_eff = k + (k − 1)(d − 1).
        k_eff_h = rng.randint(H_in // 2 if H_in // 2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in // 2 if W_in // 2 > 0 else 1, W_in)
        possible_k_d_h = ConvTranspose2DParams.find_k_d(k_eff_h)
        possible_k_d_w = ConvTranspose2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=kernel_size,
            dilation=dilation,
            padding=rng.randint(1, 4),
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_no_unit_strides_dilation(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # skip if input is 1x1, not possible to have non-unit stride tested
        if H_in == 1 and W_in == 1:
            return []
        # prepare params
        # k_eff = k + (k − 1)(d − 1) = k + kd - k - d + 1 = kd - d + 1 = d(k - 1) + 1
        k_eff_h = rng.randint(H_in // 2 if H_in // 2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in // 2 if W_in // 2 > 0 else 1, W_in)
        possible_k_d_h = ConvTranspose2DParams.find_k_d(k_eff_h)
        possible_k_d_w = ConvTranspose2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=kernel_size,
            dilation=dilation,
            stride=ConvTranspose2DParams.get_non_unit_stride_param(rng, H_in, W_in, k_eff_h, k_eff_w),
            padding=rng.randint(1, 4),
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_groups_depthwise(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        groups = ConvTranspose2DParams.get_groups_param(C_in, C_in, mode="depthwise")
        if groups == 1:
            return []  # skip: already tested in previous cases
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=C_in,
            kernel_size=ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in),
            groups=groups,
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_groups_grouped(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        C_out = rng.randint(1, C_in + 10)
        while C_out == C_in:
            C_out = rng.randint(1, C_in + 10)
        groups = ConvTranspose2DParams.get_groups_param(C_in, C_out, mode="grouped")
        if groups == 1 or groups == C_in:
            return []
        conv2DParams = ConvTranspose2DParams(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=ConvTranspose2DParams.get_kernel_param(rng, H_in, W_in),
            groups=groups,
            bias=bias,
        )
        return conv2DParams.to_kwargs_list()


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "conv_transpose_2d",  # 00
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


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    failed_killed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/test_convtranspose2d_ids_failed_killed.txt"
    )
    failed_fatal_python_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/test_convtranspose2d_ids_failed_fatal_python_error.txt"
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
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_no_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_no_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides_dilation(
                test_vector
            ),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides_dilation(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_no_unit_strides_dilation(
                test_vector
            ),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_groups_depthwise(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_groups_grouped(test_vector),
        ),
        # All tests but with no bias
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_all(test_vector, bias=False),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(
                test_vector, bias=False
            ),
            dev_data_formats=[
                item
                for item in TestCollectionData.float.dev_data_formats  # no sense to test with int data formats
                if item not in TestCollectionData.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionData.float.dev_data_formats  # no sense to test with int data formats
                if item not in TestCollectionData.single.dev_data_formats
            ],
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Test plan:
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.failed_killed,
            skip_reason=FailingReasons.INFERENCE_FROZE,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.failed_fatal_python_error,
            skip_reason=FailingReasons.FATAL_ERROR,
        ),
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.all.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# PyTorch repostiory operators


from .datatypes import OperandNumRange, OperatorDefinition, OperatorRepository, OperatorParamNumber


# TODO describe operand and shapes
_OPERATORS = [
    # nn operators
    OperatorDefinition(
        "embedding",
        "torch.nn.Embedding",
        1,
        instantiate=True,
        constructor_params=[],
    ),
    OperatorDefinition(
        "linear",
        "torch.nn.Linear",
        1,
        instantiate=True,
        constructor_params=[
            OperatorParamNumber("in_features", int, 10, 50),
            OperatorParamNumber("out_features", int, 10, 50),
        ],
    ),
    # Convolution functions
    OperatorDefinition(
        "conv2d",
        "torch.nn.Conv2d",
        1,
        instantiate=True,
        constructor_params=[
            OperatorParamNumber("in_channels", int, 10, 50),
            OperatorParamNumber("out_channels", int, 10, 50),
            OperatorParamNumber("kernel_size", int, 3, 3),
            OperatorParamNumber("stride", int, 1, 1),
            OperatorParamNumber("padding", int, 1, 1),
        ],
    ),
    OperatorDefinition(
        "conv_transpose_2d",
        "torch.nn.ConvTranspose2d",
        1,
        instantiate=True,
        constructor_params=[],
    ),
    OperatorDefinition(
        "avg_pool_2d",
        "torch.nn.AvgPool2d",
        1,
        instantiate=True,
        constructor_params=[],
    ),
    OperatorDefinition(
        "max_pool_2d",
        "torch.nn.MaxPool2d",
        1,
        instantiate=True,
        constructor_params=[],
    ),
    # Unary operators (implemented)
    OperatorDefinition("relu", "torch.relu", 1),
    OperatorDefinition("sqrt", "torch.sqrt", 1),
    OperatorDefinition("reciprocal", "torch.reciprocal", 1),
    OperatorDefinition("sigmoid", "torch.sigmoid", 1),
    OperatorDefinition("abs", "torch.abs", 1),
    OperatorDefinition("cos", "torch.cos", 1),
    OperatorDefinition("exp", "torch.exp", 1),
    OperatorDefinition("neg", "torch.neg", 1),
    OperatorDefinition("rsqrt", "torch.rsqrt", 1),
    OperatorDefinition("sin", "torch.sin", 1),
    OperatorDefinition("square", "torch.square", 1),
    OperatorDefinition(
        "pow",
        "torch.pow",
        1,
        forward_params=[
            OperatorParamNumber("exponent", int, -10, 10),
        ],
    ),
    OperatorDefinition(
        "clamp",
        "torch.clamp",
        1,
        forward_params=[
            OperatorParamNumber("min", int, -100, 100),
            OperatorParamNumber("max", int, -100, 100),
        ],
    ),
    OperatorDefinition("log", "torch.log", 1),
    OperatorDefinition("log1p", "torch.log1p", 1),
    OperatorDefinition("gelu", "torch.nn.functional.gelu", 1),
    OperatorDefinition("leaky_relu", "torch.nn.functional.leaky_relu", 1),
    OperatorDefinition("logical_not", "torch.logical_not", 1),
    OperatorDefinition(
        "cumsum",
        "torch.cumsum",
        1,
        forward_params=[
            OperatorParamNumber("dim", int, -3, 3),
        ],
    ),
    OperatorDefinition("softmax", "torch.softmax", 1),
    # Unary operators (not implemented)
    OperatorDefinition("acos", "torch.acos", 1),
    OperatorDefinition("acosh", "torch.acosh", 1),
    OperatorDefinition("angle", "torch.angle", 1),
    OperatorDefinition("asin", "torch.asin", 1),
    OperatorDefinition("asinh", "torch.asinh", 1),
    OperatorDefinition("atan", "torch.atan", 1),
    OperatorDefinition("atanh", "torch.atanh", 1),
    OperatorDefinition("bitwise_not", "torch.bitwise_not", 1),
    OperatorDefinition("ceil", "torch.ceil", 1),
    OperatorDefinition("conj_physical", "torch.conj_physical", 1),
    OperatorDefinition("cosh", "torch.cosh", 1),
    OperatorDefinition("deg2rad", "torch.deg2rad", 1),
    OperatorDefinition("digamma", "torch.digamma", 1),
    OperatorDefinition("erf", "torch.erf", 1),
    OperatorDefinition("erfc", "torch.erfc", 1),
    OperatorDefinition("erfinv", "torch.erfinv", 1),
    OperatorDefinition("exp2", "torch.exp2", 1),
    OperatorDefinition("expm1", "torch.expm1", 1),
    OperatorDefinition("floor", "torch.floor", 1),
    OperatorDefinition("frac", "torch.frac", 1),
    OperatorDefinition("lgamma", "torch.lgamma", 1),
    OperatorDefinition("log10", "torch.log10", 1),
    OperatorDefinition("log2", "torch.log2", 1),
    OperatorDefinition("logit", "torch.logit", 1),
    OperatorDefinition("i0", "torch.i0", 1),
    OperatorDefinition("isnan", "torch.isnan", 1),
    OperatorDefinition("nan_to_num", "torch.nan_to_num", 1),
    OperatorDefinition("positive", "torch.positive", 1),
    OperatorDefinition("rad2deg", "torch.rad2deg", 1),
    OperatorDefinition("round", "torch.round", 1),
    OperatorDefinition("sign", "torch.sign", 1),
    OperatorDefinition("sgn", "torch.sgn", 1),
    OperatorDefinition("signbit", "torch.signbit", 1),
    OperatorDefinition("sinc", "torch.sinc", 1),
    OperatorDefinition("sinh", "torch.sinh", 1),
    OperatorDefinition("tan", "torch.tan", 1),
    OperatorDefinition("tanh", "torch.tanh", 1),
    OperatorDefinition("trunc", "torch.trunc", 1),
    OperatorDefinition(
        "nop",
        "torch.nn.Identity",
        1,
        instantiate=True,
        constructor_params=[],
    ),
    # Binary operators
    OperatorDefinition("add", "torch.add", 2),
    OperatorDefinition("sub", "torch.sub", 2),
    OperatorDefinition("mul", "torch.mul", 2),
    OperatorDefinition("div", "torch.div", 2),
    OperatorDefinition("ge", "torch.ge", 2),
    OperatorDefinition("ne", "torch.ne", 2),
    OperatorDefinition("gt", "torch.gt", 2),
    OperatorDefinition("lt", "torch.lt", 2),
    OperatorDefinition("maximum", "torch.maximum", 2),
    OperatorDefinition("minimum", "torch.minimum", 2),
    OperatorDefinition("heaviside", "torch.heaviside", 2),
    # Binary operators (not implemented)
    OperatorDefinition("atan2", "torch.atan2", 2),
    OperatorDefinition("bitwise_and", "torch.bitwise_and", 2),
    OperatorDefinition("bitwise_or", "torch.bitwise_or", 2),
    OperatorDefinition("bitwise_xor", "torch.bitwise_xor", 2),
    OperatorDefinition("bitwise_left_shift", "torch.bitwise_left_shift", 2),
    OperatorDefinition("bitwise_right_shift", "torch.bitwise_right_shift", 2),
    OperatorDefinition("floor_divide", "torch.floor_divide", 2),
    OperatorDefinition("fmod", "torch.fmod", 2),
    OperatorDefinition("logaddexp", "torch.logaddexp", 2),
    OperatorDefinition("logaddexp2", "torch.logaddexp2", 2),
    OperatorDefinition("nextafter", "torch.nextafter", 2),
    OperatorDefinition("remainder", "torch.remainder", 2),
    OperatorDefinition("fmax", "torch.fmax", 2),
    OperatorDefinition("fmin", "torch.fmin", 2),
    OperatorDefinition("eq", "torch.eq", 2),
    OperatorDefinition("le", "torch.le", 2),
    # Matmul
    OperatorDefinition("matmul", "torch.matmul", 2),
    # Nary operators
    OperatorDefinition(
        "concatenate",
        "torch.concatenate",
        input_num_range=(2, 7),
        forward_params=[
            OperatorParamNumber("dim", int, -10, 10),
        ],
    ),
    OperatorDefinition("where", "torch.where", 3),
    # Reduce operators
    OperatorDefinition("max", "torch.max", 1),
    OperatorDefinition("sum", "torch.sum", 1),
    OperatorDefinition("mean", "torch.mean", 1),
    # TM operators
    OperatorDefinition("repeat_interleave", "torch.repeat_interleave", 1),
    OperatorDefinition("reshape", "torch.reshape", 1),
    OperatorDefinition("squeeze", "torch.squeeze", 1),
    OperatorDefinition("unsqueeze", "torch.unsqueeze", 1),
    OperatorDefinition("transpose", "torch.transpose", 1),
    OperatorDefinition(
        "stack",
        "torch.stack",
        (2, 4),
        forward_params=[
            OperatorParamNumber("dim", int, 1, 10),
        ],
    ),
    # Activation functions
    OperatorDefinition("layer_norm", "torch.nn.LayerNorm", 1),
    OperatorDefinition("batch_norm_1d", "torch.nn.BatchNorm1d", 1),
    OperatorDefinition("batch_norm_2d", "torch.nn.BatchNorm2d", 1),
    OperatorDefinition("batch_norm_3d", "torch.nn.BatchNorm3d", 1),
]


pytorch_operator_repository = OperatorRepository([op for op in _OPERATORS])

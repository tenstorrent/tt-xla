# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Forge repostiory operators


from .datatypes import OperatorDefinition, OperatorRepository
from .datatypes import OperatorParamNumber


# TODO describe operand and shapes
_OPERATORS = [
    # Convolution functions
    OperatorDefinition(
        "conv2d",
        "forge.op.Conv2d",
        3,
        forward_params=[],
    ),
    OperatorDefinition(
        "conv_transpose_2d",
        "forge.op.Conv2dTranspose",
        3,
        forward_params=[],
    ),
    # Unary operators
    OperatorDefinition("exp", "forge.op.Exp", 1),
    OperatorDefinition("reciprocal", "forge.op.Reciprocal", 1),
    OperatorDefinition("sqrt", "forge.op.Sqrt", 1),
    OperatorDefinition("relu", "forge.op.Relu", 1),
    OperatorDefinition(
        "leaky_relu",
        "forge.op.LeakyRelu",
        1,
        forward_params=[
            OperatorParamNumber("alpha", float, 0, 100),
        ],
    ),
    OperatorDefinition("nop", "forge.op.Identity", 1),
    OperatorDefinition("gelu", "forge.op.Gelu", 1),
    OperatorDefinition("log", "forge.op.Log", 1),
    OperatorDefinition("sigmoid", "forge.op.Sigmoid", 1),
    OperatorDefinition(
        "clip",
        "forge.op.Clip",
        1,
        forward_params=[
            OperatorParamNumber("min", float, 0, 100),
            OperatorParamNumber("max", float, 0, 100),
        ],
    ),
    OperatorDefinition("atan", "forge.op.Atan", 1),
    OperatorDefinition("sine", "forge.op.Sine", 1),
    OperatorDefinition("cosine", "forge.op.Cosine", 1),
    OperatorDefinition("abs", "forge.op.Abs", 1),
    OperatorDefinition("tanh", "forge.op.Tanh", 1),
    OperatorDefinition(
        "cumsum",
        "forge.op.CumSum",
        1,
        forward_params=[
            OperatorParamNumber("dim", int, -3, 3),
        ],
    ),
    OperatorDefinition("argmax", "forge.op.Argmax", 1),
    OperatorDefinition("logical_not", "forge.op.LogicalNot", 1),
    OperatorDefinition("dropout", "forge.op.Dropout", 1),
    OperatorDefinition(
        "pow",
        "forge.op.Pow",
        1,
        forward_params=[
            OperatorParamNumber("exponent", float, 0, 100),
        ],
    ),
    OperatorDefinition("erf", "forge.op.Erf", 1),
    # Binary operators
    OperatorDefinition("add", "forge.op.Add", 2),
    OperatorDefinition("divide", "forge.op.Divide", 2),
    OperatorDefinition("remainder", "forge.op.Remainder", 2),
    OperatorDefinition("subtract", "forge.op.Subtract", 2),
    OperatorDefinition("multiply", "forge.op.Multiply", 2),
    OperatorDefinition("maximum", "forge.op.Max", 2),
    OperatorDefinition("minimum", "forge.op.Min", 2),
    OperatorDefinition("heaviside", "forge.op.Heaviside", 2),
    OperatorDefinition("power", "forge.op.Power", 2),
    OperatorDefinition("greater", "forge.op.Greater", 2),
    OperatorDefinition("greater_equal", "forge.op.GreaterEqual", 2),
    OperatorDefinition("less", "forge.op.Less", 2),
    OperatorDefinition("less_equal", "forge.op.LessEqual", 2),
    OperatorDefinition("equal", "forge.op.Equal", 2),
    OperatorDefinition("not_equal", "forge.op.NotEqual", 2),
    OperatorDefinition("logical_and", "forge.op.LogicalAnd", 2),
    # Nary operators
    OperatorDefinition("where", "forge.op.Where", 3),
    # OperatorDefinition("index_copy", "forge.op.IndexCopy", 3),  # Bug #2705
    OperatorDefinition(
        "interleave",
        "forge.op.Interleave",
        (1, 10),
        forward_params=[
            OperatorParamNumber("axis", int, -3, -3),
            OperatorParamNumber("stride", int, 1, 1),
        ],
    ),
    OperatorDefinition(
        "concatenate",
        "forge.op.Concatenate",
        (1, 10),
        forward_params=[
            OperatorParamNumber("axis", int, -10, 10),
        ],
    ),
    OperatorDefinition(
        "stack",
        "forge.op.Stack",
        (2, 4),
        forward_params=[
            OperatorParamNumber("axis", int, 1, 10),
        ],
    ),
    OperatorDefinition("matmul", "forge.op.Matmul", 2),
    # OperatorDefinition("sparse_matmul", "forge.op.SparseMatmul", 2),
    # TM operators
    OperatorDefinition("repeat_interleave", "forge.op.RepeatInterleave", 1),
    OperatorDefinition("reshape", "forge.op.Reshape", 1),
    OperatorDefinition(
        "squeeze",
        "forge.op.Squeeze",
        1,
        forward_params=[
            OperatorParamNumber("dim", int, -3, 3),
        ],
    ),
    OperatorDefinition(
        "unsqueeze",
        "forge.op.Unsqueeze",
        1,
        forward_params=[
            OperatorParamNumber("dim", int, -3, 3),
        ],
    ),
    OperatorDefinition("transpose", "forge.op.Transpose", 1),
    # Activation functions
    OperatorDefinition("layer_norm", "forge.op.Layernorm", 1),
]


forge_operator_repository = OperatorRepository([op for op in _OPERATORS])

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator repository datatypes
#
# Central place for defining all Forge, PyTorch, ... operators
#
# Usage of repository:
#  - RGG (Random Graph Generator)
#  - Single operator tests
#  - TVM python_codegen.py


from .datatypes import OperandNumInt, OperandNumTuple, OperandNumRange
from .datatypes import TensorShape, OperatorParam, OperatorParamNumber, OperatorDefinition, OperatorRepository
from .datatypes import ShapeCalculationContext
from .forge_operators import forge_operator_repository
from .pytorch_operators import pytorch_operator_repository

__ALL__ = [
    "OperandNumInt",
    "OperandNumTuple",
    "OperandNumRange",
    "TensorShape",
    "OperatorParam",
    "OperatorParamNumber",
    "OperatorDefinition",
    "OperatorRepository",
    "ShapeCalculationContext",
    "forge_operator_repository",
    "pytorch_operator_repository",
]

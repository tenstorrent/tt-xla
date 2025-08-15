# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator repository models


from random import Random
from typing import List, Dict, Tuple, Optional, Callable, Type, Union
from dataclasses import dataclass, field


# Defining a type for tensor shape
TensorShape = Tuple[int, ...]


@dataclass
class OperatorParamNumber:
    """
    Define parameters of numeric type for operators

    Attributes:
        name: The name of the parameter
        type: The type of the parameter, either int or float
        min_value: The minimum value of the parameter
        max_value: The maximum value of the parameter

    """

    name: str
    type: Type[Union[int, float]]
    min_value: Optional[int]
    max_value: Optional[int]


# Define a type for operator parameters
# Later it could be extended to support more types
OperatorParam = Union[OperatorParamNumber]


OperandNumInt = int
OperandNumTuple = Tuple[int, int]


@dataclass
class OperandNumRange:
    """
    Define the range of number of operands for an operator with variable number of operands

    Attributes:
        operands_min: The minimum number of operands
        operands_max: The maximum number of operands

    """

    operands_min: int
    operands_max: int


@dataclass
class OperatorDefinition:
    """
    Definition of an operator

    Attributes:
        name: The name of the operator. E.g. "exp"
        full_name: The full class name of the operator including the module. E.g. "forge.op.Exp" or "torch.exp"
        input_num_range: Number of input operands, could be a single number, a tuple, or a range for variable number of operands
        instantiate: A flag to indicate if the operator needs to be instantiated in the constructor
        constructor_params: The parameters for the constructor
        forward_code: Optional custom forward code for the forward function if default is not enough. E.g. forward_code=lambda: "inputs[0].permute(0, 3, 1, 2)"
        forward_params: The parameters for the forward function
        operands: Definition of tensor operands. TBD
        calc_input_shapes: The function to calculate input shapes from output shape

    """

    name: str
    full_name: str
    input_num_range: Union[OperandNumInt, OperandNumTuple, OperandNumRange]
    instantiate: bool = False  # nn in Torch require instantiation in constructor
    constructor_params: List[OperatorParam] = field(default_factory=list)
    forward_code: Optional[Callable[[], str]] = None
    forward_params: List[OperatorParam] = field(default_factory=list)
    operands: List[str] = field(default_factory=list)  # TODO describe operand and shapes
    calc_input_shapes: Optional[
        Callable[["ShapeCalculationContext", Random], List[TensorShape]]
    ] = None  # calculate input shapes from output shape

    def __post_init__(self):
        if isinstance(self.input_num_range, OperandNumInt):
            self.input_num_range = OperandNumRange(self.input_num_range, self.input_num_range)
        elif isinstance(self.input_num_range, Tuple):
            self.input_num_range = OperandNumRange(self.input_num_range[0], self.input_num_range[1])
        else:
            raise ValueError(f"Invalid input_num_range type {self.input_num_range}")

    @property
    def is_operator(self) -> bool:
        return not self.instantiate

    @property
    def is_layer(self) -> bool:
        return self.instantiate


class ShapeCalculationContext:
    """
    Interface of a context object for calculating input shapes
    """

    @property
    def operator(self) -> OperatorDefinition:
        raise NotImplementedError("Operator is not defined")

    @property
    def input_num(self) -> int:
        raise NotImplementedError("input_num is not defined")

    @property
    def constructor_kwargs(self) -> Dict[str, object]:
        raise NotImplementedError("constructor_kwargs is not defined")

    @property
    def forward_kwargs(self) -> Dict[str, object]:
        raise NotImplementedError("forward_kwargs is not defined")

    @property
    def output_shape(self) -> TensorShape:
        raise NotImplementedError("output_shape is not defined")

    @property
    def rng_shape(self) -> Random:
        raise NotImplementedError("rng_shape is not defined")


class OperatorRepository:
    """
    Define a collection of operators for a specific framework

    Attributes:
        operators: The collection of all supported operators
    """

    def __init__(self, operators: List[OperatorDefinition]):
        self.operators = operators
        # Check if operator names which are unique
        names = [op.name for op in operators]
        duplicates = set([name for name in names if names.count(name) > 1])
        if duplicates:
            raise ValueError(f"Detected duplicate operator names: {duplicates}")

    def get_by_name(self, name: str):
        operators = [op for op in self.operators if op.name == name]
        if len(operators) == 1:
            return operators[0]
        if len(operators) == 0:
            raise ValueError(f"Operator {name} not found in operator repository")
        if len(operators) > 1:
            raise ValueError(
                f"Ambiguous operator name {name}, found {len(operators)} candidates in operator repository"
            )

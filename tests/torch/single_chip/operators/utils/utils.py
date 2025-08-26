# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator test utilities

import random
import sys
import torch
import pytest


from loguru import logger
from typing import Optional, List, Type, Union


from forge.op_repo.pytorch_operators import pytorch_operator_repository

from .compat import create_torch_inputs

from .datatypes import (
    TensorShape,
    AutomaticValueChecker,
    AllCloseValueChecker,
    ValueRange,
    ValueRanges,
    OperatorParameterTypes,
    FrameworkDataFormat,
)


class ShapeUtils:
    @staticmethod
    def switch_last_two(t):
        if len(t) < 2:
            return t  # If tuple has less than 2 elements, return it as is
        return t[:-2] + (t[-1], t[-2])

    @staticmethod
    def reduce_microbatch_size(shape: TensorShape) -> TensorShape:
        """
        Reduce microbatch dimension of a shape to 1
        Usually used for calculating shape of a constant tensor
        """
        return (1,) + shape[1:]

    @staticmethod
    def extend_shapes_with_id(shapes: List[TensorShape]) -> List[TensorShape]:
        """
        Extend shapes with an id
        """
        shapes_with_ids = list()
        for shape in shapes:
            if type(shape) is tuple:
                shapes_with_ids.append(pytest.param(shape, marks=tuple(), id=f"shape={shape}"))
            else:
                shapes_with_ids.append(pytest.param(shape.values[0], marks=shape.marks, id=f"shape={shape.values[0]}"))
        return shapes_with_ids


class TensorUtils:
    def create_torch_constant(
        input_shape: TensorShape,
        reduce_microbatch: bool = False,
        dev_data_format: FrameworkDataFormat = None,
        value_range: Optional[Union[ValueRanges, ValueRange, OperatorParameterTypes.RangeValue]] = None,
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        input_shape = ShapeUtils.reduce_microbatch_size(input_shape) if reduce_microbatch else input_shape

        constant = create_torch_inputs([input_shape], dev_data_format, value_range, random_seed)[0]

        return constant


class ValueCheckerUtils:
    """Utility functions for value checking"""

    @staticmethod
    def automatic(pcc: float = 0.99, rtol: float = 1e-05, atol: float = 1e-08, dissimilarity_threshold: float = 1e-03):
        return AutomaticValueChecker(pcc=pcc, rtol=rtol, atol=atol, dissimilarity_threshold=dissimilarity_threshold)

    @staticmethod
    def all_close(rtol: float = 1e-05, atol: float = 1e-08):
        return AllCloseValueChecker(rtol=rtol, atol=atol)


class LoggerUtils:
    """Utility functions for logging"""

    @staticmethod
    def set_log_level(package_name: str, level: str):
        """Set log level for package_name and its subpackages

        Args:
            package_name (str): package name
            level (str): log level
        """
        logger.add(sys.stdout, level=level, filter=lambda record: record["name"].startswith(package_name))


class RateLimiter:
    """Rate limiter class to limit the number of allowed operations by a rate limit factor"""

    def __init__(self, rng: random.Random, max_limit: int, current_limit: int):
        self.rng = rng
        self.max_limit = max_limit
        self.current_limit = current_limit
        self.current_value: int = None

    def is_allowed(self) -> bool:
        """Check if the operation is allowed by the rate limit factor and current random value"""
        self.current_value = self.rng.randint(1, self.max_limit)
        return self.current_value <= self.current_limit

    def limit_info(self) -> str:
        """Return the rate limit info for previous operation"""
        if self.current_value < self.current_limit:
            return f"{self.current_value} <= {self.current_limit}"
        else:
            return f"{self.current_value} > {self.current_limit}"


class PytorchUtils:
    """Utility functions for PyTorch operators"""

    @staticmethod
    def get_op_class_by_name(op_name: str) -> Type:
        """Get class name of the given operator"""
        # Get the module that contains the operator
        op_full_name = pytorch_operator_repository.get_by_name(op_name).full_name
        module_name = op_full_name.rsplit(".", 1)[0]
        ## module = importlib.import_module(module_name)  # bad performance
        module = torch
        if module_name == "torch.nn.functional":
            module = torch.nn.functional
        if module_name == "torch.nn":
            module = torch.nn
        # Get operator name from full name
        name = op_full_name.rsplit(".", 1)[-1]
        # Get the operator class from the module
        op_class = getattr(module, name)
        return op_class

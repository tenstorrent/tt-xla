# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator test utilities

import random
import os
import sys
import torch
import pytest


from enum import Enum
from dataclasses import dataclass
from loguru import logger
from typing import Optional, List, Dict, Type, Union

from .forge import ForgeModule, Module
from .forge import DataFormat
from .forge import TensorShape
from .forge import pytorch_operator_repository
from .forge import MathFidelity

from .forge import CompilerConfig
from .forge import VerifyConfig
from .forge import ValueChecker
from .forge import AutomaticValueChecker
from .forge import AllCloseValueChecker

from .compat import TestDevice
from .compat import (
    create_torch_inputs,
    verify_module_for_inputs,
    verify_module_for_inputs_torch,
)
from .datatypes import ValueRange, ValueRanges
from .datatypes import OperatorParameterTypes
from .datatypes import FrameworkDataFormat
from .features import TestSweepsFeatures


# All supported framework model types
FrameworkModelType = Union[
    Type[torch.nn.Module],
]


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


@dataclass(frozen=True)
class InputSourceFlag:
    """Dataclass for specifying compiler flags for specific input source"""

    input_queues_on_host: bool
    set_default_dram_parameters: bool
    default_dram_parameters: Optional[bool]


class InputSourceFlags(Enum):
    """Enums defining input source flags"""

    FROM_HOST = InputSourceFlag(True, False, None)
    FROM_DRAM = InputSourceFlag(False, False, None)
    FROM_DRAM_PROLOGUED = InputSourceFlag(False, True, False)
    FROM_DRAM_NOT_PROLOGUED = InputSourceFlag(False, True, True)
    FROM_DRAM_PROLOGUE_MICROBATCH_SIZE = InputSourceFlag(False, True, None)


class CompilerUtils:
    """Utility functions for Forge compiler configuration"""

    @staticmethod
    def set_input_source(input_source_flag: InputSourceFlag, compiler_cfg: CompilerConfig):
        """Set compiler configuration for input source"""
        # Not existing in the compiler, after global config removal
        # compiler_cfg.input_queues_on_host = input_source_flag.input_queues_on_host
        # if input_source_flag.set_default_dram_parameters:
        #     compiler_cfg.default_dram_parameters = input_source_flag.default_dram_parameters

        # NOP since we don't use this flag in the compiler, currently.

    @staticmethod
    def set_math_fidelity(math_fidelity: MathFidelity, compiler_cfg: CompilerConfig):
        """Set compiler configuration for math fidelity"""
        # Currently not respected/supported in the compiler
        compiler_cfg.default_math_fidelity = math_fidelity


class DeviceUtils:
    """Utility functions for Forge verification"""

    @staticmethod
    def warm_reset():
        reset_command = "/home/software/syseng/wh/tt-smi -lr all wait -er"
        os.system(reset_command)


class ValueCheckerUtils:
    """Utility functions for value checking"""

    @staticmethod
    def automatic(pcc: float = 0.99, rtol: float = 1e-05, atol: float = 1e-08, dissimilarity_threshold: float = 1e-03):
        return AutomaticValueChecker(pcc=pcc, rtol=rtol, atol=atol, dissimilarity_threshold=dissimilarity_threshold)

    @staticmethod
    def all_close(rtol: float = 1e-05, atol: float = 1e-08):
        return AllCloseValueChecker(rtol=rtol, atol=atol)


class VerifyUtils:
    """Utility functions for Forge verification"""

    @classmethod
    def verify(
        cls,
        model: Module,
        test_device: TestDevice,
        input_shapes: List[TensorShape],
        input_params: List[Dict] = [],
        model_dtype: Optional[torch.dtype] = None,
        compiler_cfg: Optional[CompilerConfig] = None,  # TODO Remove obsoleted
        pcc: Optional[float] = None,
        input_source_flag: InputSourceFlags = None,
        dev_data_format: DataFormat = None,
        convert_to_forge: Optional[bool] = None,
        math_fidelity: MathFidelity = None,
        value_range: Optional[ValueRanges] = None,
        random_seed: Optional[int] = None,
        warm_reset: bool = False,
        value_checker: Optional[ValueChecker] = None,
        verify_config: Optional[VerifyConfig] = None,  # TODO Remove obsoleted
        skip_forge_verification: bool = TestSweepsFeatures.params.skip_forge_verification,
    ):
        """Perform Forge verification on the model

        Args:
            model: Forge model
            test_device: TestDevice
            input_shapes: List of input shapes
            input_params: List of input parameters
            model_dtype: Model data type to transfer model to
            compiler_cfg: Compiler configuration
            pcc: PCC value for verification
            input_source_flag: Input source flag
            dev_data_format: Data format
            convert_to_forge: Convert input tensors to Forge data format
            math_fidelity: Math fidelity
            value_range: Value range of input tensors
            random_seed: Random seed
            warm_reset: Warm reset the device before verification
            value_checker: Value checker for verification
            verify_config: Verification configuration
            skip_forge_verification: Skip verification with Forge module
        """

        if not compiler_cfg:
            compiler_cfg = CompilerConfig()

        if not verify_config:
            verify_config = VerifyConfig()

        if value_checker:
            verify_config.value_checker = value_checker

        if model_dtype:
            # Transfer model to model_dtype if specified
            model.to(model_dtype)
            if model_dtype is torch.bfloat16:
                # Override default data format for bfloat16
                compiler_cfg.default_df_override = DataFormat.Float16_b

        # Conclude if we should convert to forge data format
        if convert_to_forge is None and isinstance(model, ForgeModule):
            convert_to_forge = True

        cls.setup(
            compiler_cfg=compiler_cfg,
            input_source_flag=input_source_flag,
            math_fidelity=math_fidelity,
            warm_reset=warm_reset,
        )

        inputs = cls.create_torch_inputs(
            input_shapes=input_shapes,
            dev_data_format=dev_data_format,
            value_range=value_range,
            random_seed=random_seed,
        )

        cls.verify_module_for_inputs(
            model=model,
            inputs=inputs,
            compiler_cfg=compiler_cfg,
            pcc=pcc,
            verify_config=verify_config,
            dev_data_format=dev_data_format,
            convert_to_forge=convert_to_forge,
            skip_forge_verification=skip_forge_verification,
        )

    @classmethod
    def setup(
        cls,
        compiler_cfg: CompilerConfig,
        input_source_flag: InputSourceFlags = None,
        math_fidelity: MathFidelity = None,
        warm_reset: bool = False,
    ):
        if warm_reset:
            DeviceUtils.warm_reset()

        if input_source_flag:
            CompilerUtils.set_input_source(input_source_flag.value, compiler_cfg)

        if math_fidelity:
            CompilerUtils.set_math_fidelity(math_fidelity, compiler_cfg)

        # if dev_data_format:
        #     input_params.append({"dev_data_format": dev_data_format})

    @classmethod
    def create_torch_inputs(
        cls,
        input_shapes: List[TensorShape],
        dev_data_format: DataFormat = None,
        value_range: Optional[ValueRanges] = None,
        random_seed: Optional[int] = None,
    ) -> List[torch.Tensor]:

        inputs = create_torch_inputs(
            input_shapes=input_shapes,
            dev_data_format=dev_data_format,
            value_range=value_range,
            random_seed=random_seed,
        )

        return inputs

    @classmethod
    def verify_module_for_inputs(
        cls,
        model: Module,
        inputs: List[torch.Tensor],
        compiler_cfg: CompilerConfig,
        pcc: Optional[float] = None,
        verify_config: Optional[VerifyConfig] = None,
        dev_data_format: DataFormat = None,
        convert_to_forge: bool = True,  # explicit conversion to forge data format
        skip_forge_verification: bool = TestSweepsFeatures.params.skip_forge_verification,
    ):

        if skip_forge_verification:
            if isinstance(model, ForgeModule):
                logger.warning("Nothing to validate while skipping Forge verification for Forge module")
            else:
                verify_module_for_inputs_torch(
                    model=model,
                    inputs=inputs,
                    verify_config=verify_config,
                )
        else:
            verify_module_for_inputs(
                model=model,
                inputs=inputs,
                compiler_cfg=compiler_cfg,
                verify_config=verify_config,
                dev_data_format=dev_data_format,
                convert_to_forge=convert_to_forge,
            )


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

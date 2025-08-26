# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Datatypes for operator test utilities

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Union, Tuple, Type, TypeAlias, Any

import torch

from forge.op_repo.datatypes import TensorShape

# from .forge import ForgeTensor

# ForgeDataFormat = DataFormat
# ForgeMathFidelity = MathFidelity


@dataclass
class DataFormatData:
    name: str


class DataFormat(Enum):
    Float32 = DataFormatData(name="Float32")
    Float16 = DataFormatData(name="Float16")
    Bfp8 = DataFormatData(name="Bfp8")
    Bfp4 = DataFormatData(name="Bfp4")
    Bfp2 = DataFormatData(name="Bfp2")
    Float16_b = DataFormatData(name="Float16_b")
    Bfp8_b = DataFormatData(name="Bfp8_b")
    Bfp4_b = DataFormatData(name="Bfp4_b")
    Bfp2_b = DataFormatData(name="Bfp2_b")
    Lf8 = DataFormatData(name="Lf8")
    UInt16 = DataFormatData(name="UInt16")
    Int8 = DataFormatData(name="Int8")
    Int32 = DataFormatData(name="Int32")
    RawUInt8 = DataFormatData(name="RawUInt8")
    RawUInt16 = DataFormatData(name="RawUInt16")
    RawUInt32 = DataFormatData(name="RawUInt32")


class MathFidelity(Enum):
    LoFi = DataFormatData(name="LoFi")
    HiFi2 = DataFormatData(name="HiFi2")
    HiFi3 = DataFormatData(name="HiFi3")
    HiFi4 = DataFormatData(name="HiFi4")


FrameworkDataFormat = Union[DataFormat, torch.dtype]


class OperatorParameterTypes:
    SingleValue: TypeAlias = Union[int, float]
    RangeValue: TypeAlias = Tuple[SingleValue, SingleValue]
    Value: TypeAlias = Union[SingleValue, RangeValue]
    Kwargs: TypeAlias = Dict[str, Value]


# All supported framework model types
FrameworkModelType = Union[
    Type[torch.nn.Module],
    # Module,
    Any,
]


# TODO - Remove this mock class once TestDevice is available in Forge
# https://github.com/tenstorrent/tt-forge-fe/issues/342
class TestDevice:

    pass


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


@dataclass(frozen=True)
class ValueRange:
    """Dataclass for specifying compiler flags for specific input source"""

    low: Optional[OperatorParameterTypes.SingleValue]
    high: Optional[OperatorParameterTypes.SingleValue]

    def get_range(self) -> OperatorParameterTypes.RangeValue:
        """Get the range values"""
        return self.low, self.high


class ValueRanges(Enum):
    """Enums defining value ranges"""

    SMALL = ValueRange(-1, 1)
    SMALL_POSITIVE = ValueRange(0, 1)
    SMALL_NEGATIVE = ValueRange(-1, 0)
    LARGE = ValueRange(None, None)
    LARGE_POSITIVE = ValueRange(0, None)
    LARGE_NEGATIVE = ValueRange(None, 0)


@dataclass
class ValueChecker:
    pass


@dataclass
class AutomaticValueChecker(ValueChecker):
    pcc: float = 0.99
    rtol: float = 1e-05
    atol: float = 1e-08
    dissimilarity_threshold: float = 1e-03


@dataclass
class AllCloseValueChecker(ValueChecker):
    rtol: float = 1e-05
    atol: float = 1e-08


@dataclass
class VerifyConfig:
    """
    Dataclass for configuration used in model verification.

    Attributes:
        model (Any): Test model to be verified.
        input_shapes (List[Any]): List of input shapes for the model input.
        inputs (Optional[List[torch.Tensor]]): List of input tensors for the model.
        test_device (Optional[Any]): Device on which to run the test.
        model_dtype (Optional[torch.dtype]): Data type to which model will be transferred.
        input_source_flag (Optional[InputSourceFlags]): Flags indicating the source of input data.
        dev_data_format (Optional[DataFormat]): Data format used on the device.
        convert_to_forge (Optional[bool]): Whether to convert input tensors to Forge tensors.
        math_fidelity (Optional[MathFidelity]): Math fidelity settings for verification.
        value_range (Optional[ValueRanges]): Value range of input tensors.
        random_seed (Optional[int]): Random seed.
        warm_reset (Optional[bool]): Whether to perform a warm reset before verification.
        value_checker (Optional[ValueChecker]): Custom value checker for output validation.
        skip_forge_verification (Optional[bool]): If True, skips Forge verification.
        enabled (bool): Enables or disables verification.
    """

    model: FrameworkModelType
    input_shapes: List[TensorShape]
    inputs: Optional[List[torch.Tensor]] = None
    test_device: Optional[TestDevice] = None  # TODO remove obsoleted
    model_dtype: Optional[torch.dtype] = None
    input_source_flag: Optional[InputSourceFlags] = None
    dev_data_format: Optional[DataFormat] = None
    convert_to_forge: Optional[bool] = None
    math_fidelity: Optional[MathFidelity] = None
    value_range: Optional[ValueRanges] = None
    random_seed: Optional[int] = None
    warm_reset: Optional[bool] = False
    value_checker: Optional[ValueChecker] = None
    skip_forge_verification: Optional[bool] = None

    enabled: bool = True  # enable/disable verification

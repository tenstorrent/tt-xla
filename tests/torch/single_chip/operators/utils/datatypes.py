# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Datatypes for operator test utilities

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Union, Tuple, TypeAlias

import torch
from .forge import DataFormat

# from .forge import MathFidelity
# from .forge import ForgeTensor

# ForgeDataFormat = DataFormat
# ForgeMathFidelity = MathFidelity

FrameworkDataFormat = Union[DataFormat, torch.dtype]


class OperatorParameterTypes:
    SingleValue: TypeAlias = Union[int, float]
    RangeValue: TypeAlias = Tuple[SingleValue, SingleValue]
    Value: TypeAlias = Union[SingleValue, RangeValue]
    Kwargs: TypeAlias = Dict[str, Value]


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

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Compatibility methods and datatypes for Forge

import torch

from loguru import logger
from typing import Optional, List, Union

from .frontend import SweepsFrontend

from .frontend.datatypes import ForgeTensor

from .datatypes import (
    DataFormat,
    FrameworkDataFormat,
    TensorShape,
    OperatorParameterTypes,
    ValueRanges,
    ValueRange,
    VerifyConfig,
)


class TestTensorsUtils:
    """Utility class for generating random torch tensors"""

    # The map defines which torch dtype is used while testing for a given dev data format
    # The torch dtype can be smaller or bigger than the dev data format
    # Generated random values are chosen to fit both the torch dtype and the dev data format
    # So after the conversion from torch to forge no data is lost
    dev_data_format_to_dtype = {
        # DataFormat.Bfp2: torch.float16,
        # DataFormat.Bfp2: torch.bfloat16,
        DataFormat.Bfp2: torch.float32,
        DataFormat.Bfp2_b: torch.float32,
        # DataFormat.Bfp2_b: torch.bool,  # mapped
        DataFormat.Bfp4: torch.float32,
        DataFormat.Bfp4_b: torch.float32,
        DataFormat.Bfp8: torch.float32,
        DataFormat.Bfp8_b: torch.float32,
        # DataFormat.Float16: torch.float16,  # fatal python error
        DataFormat.Float16: torch.float32,  # mapped
        DataFormat.Float16_b: torch.float32,  # mapped
        DataFormat.Float32: torch.float32,  # mapped
        # DataFormat.Lf8: torch.float16,  # fatal python error
        # DataFormat.Lf8: torch.bfloat16,  # pcc failed
        DataFormat.Lf8: torch.float32,
        # DataFormat.RawUInt8: torch.int8,  # unsupported
        # DataFormat.RawUInt8: torch.uint8,  # Unsupported torch dtype
        DataFormat.RawUInt8: torch.int32,
        # DataFormat.RawUInt16: torch.int8,
        # DataFormat.RawUInt16: torch.uint8,
        # DataFormat.RawUInt16: torch.int16,
        DataFormat.RawUInt16: torch.int32,
        # DataFormat.RawUInt32: torch.uint8,
        DataFormat.RawUInt32: torch.int32,
        # DataFormat.Int8: torch.int8,  # unsupported
        DataFormat.Int8: torch.int32,  # mapped
        # DataFormat.UInt16: torch.int8,  #  E  RuntimeError: Unsupported data type
        DataFormat.UInt16: torch.int32,
        DataFormat.Int32: torch.int32,  # mapped
    }

    # Defines ranges of values for forge data formats
    data_format_ranges = {
        DataFormat.Bfp2: (-10000, 10000),
        DataFormat.Bfp2_b: (-10000, 10000),
        DataFormat.Bfp4: (-10000, 10000),
        DataFormat.Bfp4_b: (-10000, 10000),
        DataFormat.Bfp8: (-10000, 10000),
        DataFormat.Bfp8_b: (-10000, 10000),
        DataFormat.Float16: (-10000, 10000),
        DataFormat.Float16_b: (-10000, 10000),
        DataFormat.Float32: (-10000, 10000),
        DataFormat.Lf8: (-10000, 10000),
        DataFormat.RawUInt8: (0, 2**8 - 1),
        DataFormat.RawUInt16: (0, 2**16 - 1),
        DataFormat.RawUInt32: (0, 2**32 - 1),
        DataFormat.Int8: (-(2**7), 2**7 - 1),
        DataFormat.UInt16: (0, 2**16 - 1),
        DataFormat.Int32: (-(2**31), 2**31 - 1),
    }

    # Defines ranges of values for torch dtypes
    dtype_ranges = {
        torch.bfloat16: (-10000, 10000),
        torch.float16: (-10000, 10000),
        torch.float32: (-10000, 10000),
        torch.float64: (-10000, 10000),
        torch.uint8: (0, 2**8 - 1),
        torch.int8: (-(2**7), 2**7 - 1),
        torch.int16: (-(2**15), 2**15 - 1),
        torch.int32: (-(2**31), 2**31 - 1),
        torch.int64: (-(2**63), 2**63 - 1),
    }

    class DTypes:
        """Grouping of torch dtypes"""

        floats = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
        integers = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        booleans = (torch.bool,)

    @classmethod
    def get_value_range(
        cls,
        dev_data_format: DataFormat,
        dtype: torch.dtype,
        value_range: Optional[OperatorParameterTypes.RangeValue] = None,
    ) -> OperatorParameterTypes.RangeValue:
        """Returns a range for the given data format and dtype"""

        # When dev_data_format is None, the range should be the same as Float32/float32
        if dev_data_format is None:
            dev_data_format = DataFormat.Float32
        if dtype is None:
            dtype = torch.float32

        if isinstance(dev_data_format, DataFormat):
            forge_data_format_ranges = cls.data_format_ranges
        elif isinstance(dev_data_format, torch.dtype):
            forge_data_format_ranges = cls.dtype_ranges

        if dev_data_format in forge_data_format_ranges:
            data_format_ranges = forge_data_format_ranges[dev_data_format]
        else:
            raise ValueError(f"Unsupported range for dev data format: {dev_data_format}")
        if dtype in cls.dtype_ranges:
            dtype_ranges = cls.dtype_ranges[dtype]
        else:
            raise ValueError(f"Unsupported range for dtype: {dtype}")
        range = cls.intersect_ranges(data_format_ranges, dtype_ranges)
        if value_range is not None:
            range = cls.intersect_ranges(range, value_range)
        return range

    @classmethod
    def intersect_ranges(
        cls, range1: OperatorParameterTypes.RangeValue, range2: OperatorParameterTypes.RangeValue
    ) -> OperatorParameterTypes.RangeValue:
        """Intersects two ranges"""

        low = None
        if range1[0] is None and range2[0] is None:
            low = None
        else:
            if range1[0] is None:
                low = range2[0]
            elif range2[0] is None:
                low = range1[0]
            else:
                low = max(range1[0], range2[0])

        high = None
        if range1[1] is None and range2[1] is None:
            high = None
        else:
            if range1[1] is None:
                high = range2[1]
            elif range2[1] is None:
                high = range1[1]
            else:
                high = min(range1[1], range2[1])

        return low, high

    @classmethod
    def get_dtype_for_df(cls, dev_data_format: DataFormat = None) -> torch.dtype:

        dtype: torch.dtype

        if dev_data_format is None:
            dtype = None
        elif isinstance(dev_data_format, torch.dtype):
            dtype = dev_data_format
        else:
            # dtype = torch.float32
            if dev_data_format in cls.dev_data_format_to_dtype:
                dtype = cls.dev_data_format_to_dtype[dev_data_format]
            else:
                raise ValueError(f"Unsupported dtype for dev data format: {dev_data_format}")

        return dtype

    @classmethod
    def move_from_small_to_big_value_range(cls, input: torch.Tensor, min, max) -> torch.Tensor:
        return min + (max - min) * input

    @classmethod
    def get_random_torch_input(
        cls,
        dev_data_format: DataFormat,
        dtype: torch.dtype,
        input_shape: TensorShape,
        generator: torch.Generator,
        value_range: Optional[OperatorParameterTypes.RangeValue] = None,
    ) -> List[torch.Tensor]:

        if dtype is None:
            input = torch.rand(input_shape, generator=generator)
            range = cls.get_value_range(dev_data_format=dev_data_format, dtype=dtype, value_range=value_range)
            min, max = range
            if min == 0 and max == 1:
                # No need to scale the input, just return it
                return input
            return cls.move_from_small_to_big_value_range(input, min, max)
        elif dtype in cls.DTypes.floats:
            input = torch.rand(input_shape, dtype=dtype, generator=generator)
            range = cls.get_value_range(dev_data_format=dev_data_format, dtype=dtype, value_range=value_range)
            min, max = range
            return cls.move_from_small_to_big_value_range(input, min, max)
        elif dtype in cls.DTypes.booleans:
            return torch.rand(input_shape, dtype=torch.float32, generator=generator) > 0.5
        elif dtype in cls.DTypes.integers:
            range = cls.get_value_range(dev_data_format=dev_data_format, dtype=dtype, value_range=value_range)
            low, high = range
            return torch.randint(low=low, high=high, size=input_shape, dtype=dtype, generator=generator)
        else:
            raise ValueError(f"Fail creating random torch input for unsupported dtype: {dtype}")

    @classmethod
    def extract_value_range(
        cls, value_range: Optional[Union[ValueRanges, ValueRange, OperatorParameterTypes.RangeValue]] = None
    ) -> OperatorParameterTypes.RangeValue:
        if isinstance(value_range, ValueRanges):
            value_range = value_range.value.get_range()
        elif isinstance(value_range, ValueRange):
            value_range = value_range.get_range()
        elif isinstance(value_range, tuple):
            pass
        return value_range

    @classmethod
    def convert_to_forge_tensors(cls, inputs: List[torch.Tensor], dev_data_format: DataFormat) -> List[ForgeTensor]:
        # return [ForgeTensor.create_from_torch(input, dev_data_format=dev_data_format) for input in inputs]
        return SweepsFrontend.to_forge_tensors(inputs)


# TODO move to class TestTensorsUtils
def create_torch_inputs(
    input_shapes: List[TensorShape],
    dev_data_format: FrameworkDataFormat = None,
    value_range: Optional[Union[ValueRanges, ValueRange, OperatorParameterTypes.RangeValue]] = None,
    random_seed: Optional[int] = None,
) -> List[torch.Tensor]:

    if random_seed is None:
        # Set a default seed if not provided
        random_seed = 42
    generator = torch.Generator().manual_seed(random_seed)

    dtype = TestTensorsUtils.get_dtype_for_df(dev_data_format)

    # if dtype is not None:
    #     torch.set_default_dtype(dtype)

    if value_range is not None:
        value_range = TestTensorsUtils.extract_value_range(value_range)

    inputs = [
        TestTensorsUtils.get_random_torch_input(dev_data_format, dtype, input_shape, generator, value_range)
        for input_shape in input_shapes
    ]

    return inputs


def verify_module_for_inputs(verify_config: VerifyConfig):

    error_raised = None
    try:
        SweepsFrontend.verify(verify_config=verify_config)
    except Exception as e:
        error_raised = e
        check_pcc_error_level(error_raised)
    finally:
        if not error_raised:
            pcc = SweepsFrontend.get_pcc()
            if pcc is not None and 0.99 <= pcc <= 1:
                logger.info("pcc is in valid range: 0.99 <= {} <= 1, no pcc_error raised", pcc)


def check_pcc_error_level(e: Exception):
    """
    Check the pcc error level based on the pcc value.
    """
    if "Data mismatch -> AutomaticValueChecker" in str(e):
        pcc = SweepsFrontend.get_pcc()
        if pcc:
            # logger.error(f"Original error: {e}")
            if pcc <= 0.85:
                raise ValueError(
                    f"Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid low range: {pcc} <= 0.85"
                )
            elif pcc <= 0.95:
                raise ValueError(
                    f"Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid medium range: 0.85 < {pcc} <= 0.95"
                )
            elif pcc < 0.99:
                raise ValueError(
                    f"Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid high range: 0.95 < {pcc} < 0.99"
                )
            else:
                logger.info("pcc is in valid range: 0.99 <= {} <= 1, no need to raise error", pcc)
        else:
            atol = SweepsFrontend.get_atol()
            logger.info(
                "Output is scalar so pcc wasn't calculated, it's AllClose error generated from AutomaticValueChecker, atol={}",
                atol,
            )
            raise e
    else:
        raise e

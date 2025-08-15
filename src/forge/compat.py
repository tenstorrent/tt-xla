class ForgeModule:

    pass

class Module:

    pass

class DeprecatedVerifyConfig:

    pass


# forge/forge_property_utils.py

class ForgePropertyHandler:

    @staticmethod
    def get(*args, **kwargs):
        # if "tags.pcc" in kwargs:
        #     return 1.0
        if args and len(args) > 0:
            return 1.0
        return forge_property_handler_var


forge_property_handler_var = ForgePropertyHandler()
# forge_property_handler_var.get().get("tags.pcc")


def record_sweeps_test_tags(**kwargs):
    pass


def record_sweeps_expected_failing_reason(**kwargs):
    pass


def record_sweeps_detected_failing_reason(**kwargs):
    pass


# forge/_C.py

from dataclasses import dataclass
from enum import Enum

@dataclass
class DataFormatData:
    name: str


# Float32 = 0,
# Float16 = 1,
# Bfp8 = 2,
# Bfp4 = 3,
# Bfp2 = 11,
# Float16_b = 5,
# Bfp8_b = 6,
# Bfp4_b = 7,
# Bfp2_b = 15,
# Lf8 = 10,
# UInt16 = 12,
# Int8 = 14,
# Int32 = 8,
# RawUInt8 = 0xf0,
# RawUInt16 = 0xf1,
# RawUInt32 = 0xf2,
# Invalid = 0xff



# LoFi = 0,
# HiFi2 = 2,
# HiFi3 = 3,
# HiFi4 = 4,
# Invalid = 0xff,

class DataFormat(Enum):
    # Float32 = "float32"
    # Float16 = "float16"
    # Bfp8 = "bfp8"
    # Bfp4 = "bfp4"
    # Bfp2 = "bfp2"
    # Float16_b = "float16_b"
    # Bfp8_b = "bfp8_b"
    # Bfp4_b = "bfp4_b"
    # Bfp2_b = "bfp2_b"
    # Lf8 = "lf8"
    # UInt16 = "uint16"
    # Int8 = "int8"
    # Int32 = "int32"
    # RawUInt8 = "raw_uint8"
    # RawUInt16 = "raw_uint16"
    # RawUInt32 = "raw_uint32"
    # Invalid = "invalid"
    # Float32 = 1
    # Float16 = 2
    # Bfp8 = 3
    # Bfp4 = 4
    # Bfp2 = 5
    # Float16_b = 6
    # Bfp8_b = 7
    # Bfp4_b = 8
    # Bfp2_b = 9
    # Lf8 = 10
    # UInt16 = 11
    # Int8 = 12
    # Int32 = 13
    # RawUInt8 = 14
    # RawUInt16 = 15
    # RawUInt32 = 16
    

    Float32 = DataFormatData(name="float32")
    Float16 = DataFormatData(name="float16")
    Bfp8 = DataFormatData(name="bfp8")
    Bfp4 = DataFormatData(name="bfp4")
    Bfp2 = DataFormatData(name="bfp2")
    Float16_b = DataFormatData(name="float16_b")
    Bfp8_b = DataFormatData(name="bfp8_b")
    Bfp4_b = DataFormatData(name="bfp4_b")
    Bfp2_b = DataFormatData(name="bfp2_b")
    Lf8 = DataFormatData(name="lf8")
    UInt16 = DataFormatData(name="uint16")
    Int8 = DataFormatData(name="int8")
    Int32 = DataFormatData(name="int32")
    RawUInt8 = DataFormatData(name="raw_uint8")
    RawUInt16 = DataFormatData(name="raw_uint16")
    RawUInt32 = DataFormatData(name="raw_uint32")


class MathFidelity(Enum):
    # LoFi = "lofi"
    # HiFi2 = "hifi2"
    # HiFi3 = "hifi3"
    # HiFi4 = "hifi4"
    # Invalid = "invalid"
    LoFi = DataFormatData(name="lofi")
    HiFi2 = DataFormatData(name="hifi2")
    HiFi3 = DataFormatData(name="hifi3")
    HiFi4 = DataFormatData(name="hifi4")
    # Invalid = DataFormatData(name="invalid")


# forge/config.py

class CompilerConfig:

    pass


class VerifyConfig:

    pass

# forge/tensor.py

def to_pt_tensors():
    pass


def to_forge_tensors():
    pass

# forge/__init__.py

class Tensor:
    pass


def compile(*args, **kwargs):
    pass

# forge/verify/__init__.py

class TestKind:

    pass

# forge/verify/compare.py

def compare_with_golden():
    pass

# forge/verify/config.py

class VerifyConfig:

    def __init__(self, **kwargs):
        pass


# forge/verify/value_checkers.py


class ValueChecker:
    
    def __init__(self, **kwargs):
        pass

class AllCloseValueChecker:
    
    def __init__(self, **kwargs):
        pass

class AutomaticValueChecker:

    def __init__(self, **kwargs):
        pass

# forge/verify/verify.py

def verify(*args, **kwargs):
    pass


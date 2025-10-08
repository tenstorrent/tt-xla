// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/data_type_utils.h"

// c++ standard library includes
#include <stdexcept>

// llvm mlir includes
#include "mlir/IR/BuiltinTypes.h"

namespace tt::pjrt::data_type_utils {

std::string getPJRTBufferTypeString(PJRT_Buffer_Type type) {
  switch (type) {
  case PJRT_Buffer_Type_INVALID:
    return "PJRT_Buffer_Type_INVALID";
  case PJRT_Buffer_Type_PRED:
    return "PJRT_Buffer_Type_PRED";
  case PJRT_Buffer_Type_S8:
    return "PJRT_Buffer_Type_S8";
  case PJRT_Buffer_Type_S16:
    return "PJRT_Buffer_Type_S16";
  case PJRT_Buffer_Type_S32:
    return "PJRT_Buffer_Type_S32";
  case PJRT_Buffer_Type_S64:
    return "PJRT_Buffer_Type_S64";
  case PJRT_Buffer_Type_U8:
    return "PJRT_Buffer_Type_U8";
  case PJRT_Buffer_Type_U16:
    return "PJRT_Buffer_Type_U16";
  case PJRT_Buffer_Type_U32:
    return "PJRT_Buffer_Type_U32";
  case PJRT_Buffer_Type_U64:
    return "PJRT_Buffer_Type_U64";
  case PJRT_Buffer_Type_F16:
    return "PJRT_Buffer_Type_F16";
  case PJRT_Buffer_Type_F32:
    return "PJRT_Buffer_Type_F32";
  case PJRT_Buffer_Type_F64:
    return "PJRT_Buffer_Type_F64";
  case PJRT_Buffer_Type_BF16:
    return "PJRT_Buffer_Type_BF16";
  case PJRT_Buffer_Type_C64:
    return "PJRT_Buffer_Type_C64";
  case PJRT_Buffer_Type_C128:
    return "PJRT_Buffer_Type_C128";
  case PJRT_Buffer_Type_F8E5M2:
    return "PJRT_Buffer_Type_F8E5M2";
  case PJRT_Buffer_Type_F8E4M3FN:
    return "PJRT_Buffer_Type_F8E4M3FN";
  case PJRT_Buffer_Type_F8E4M3B11FNUZ:
    return "PJRT_Buffer_Type_F8E4M3B11FNUZ";
  case PJRT_Buffer_Type_F8E5M2FNUZ:
    return "PJRT_Buffer_Type_F8E5M2FNUZ";
  case PJRT_Buffer_Type_F8E4M3FNUZ:
    return "PJRT_Buffer_Type_F8E4M3FNUZ";
  case PJRT_Buffer_Type_S4:
    return "PJRT_Buffer_Type_S4";
  case PJRT_Buffer_Type_U4:
    return "PJRT_Buffer_Type_U4";
  default:
    return "UNKNOWN_TYPE";
  }
}

PJRT_Buffer_Type
convertRuntimeToPJRTDataType(tt::target::DataType runtime_data_type) {
  switch (runtime_data_type) {
  case tt::target::DataType::UInt8:
    return PJRT_Buffer_Type_U8;
  case tt::target::DataType::UInt16:
    return PJRT_Buffer_Type_U16;
  case tt::target::DataType::UInt32:
    return PJRT_Buffer_Type_U32;
  case tt::target::DataType::Int32:
    return PJRT_Buffer_Type_S32;
  case tt::target::DataType::Float16:
    return PJRT_Buffer_Type_F16;
  case tt::target::DataType::Float32:
    return PJRT_Buffer_Type_F32;
  case tt::target::DataType::BFloat16:
    return PJRT_Buffer_Type_BF16;
  default:
    throw std::runtime_error("Unsupported runtime data type");
  }
}

tt::target::DataType
convertPJRTToRuntimeDataType(PJRT_Buffer_Type pjrt_data_type) {
  switch (pjrt_data_type) {
  case PJRT_Buffer_Type_U8:
    return tt::target::DataType::UInt8;
  case PJRT_Buffer_Type_U16:
    return tt::target::DataType::UInt16;
  case PJRT_Buffer_Type_U32:
    return tt::target::DataType::UInt32;
  case PJRT_Buffer_Type_S32:
    return tt::target::DataType::Int32;
  case PJRT_Buffer_Type_F16:
    return tt::target::DataType::Float16;
  case PJRT_Buffer_Type_F32:
    return tt::target::DataType::Float32;
  case PJRT_Buffer_Type_BF16:
    return tt::target::DataType::BFloat16;
  case PJRT_Buffer_Type_F64:
    return tt::target::DataType::Float64;
  case PJRT_Buffer_Type_S64:
    return tt::target::DataType::Int64;
  case PJRT_Buffer_Type_S16:
    return tt::target::DataType::Int16;
  case PJRT_Buffer_Type_S8:
    return tt::target::DataType::Int8;
  case PJRT_Buffer_Type_U64:
    return tt::target::DataType::UInt64;
  case PJRT_Buffer_Type_PRED:
    return tt::target::DataType::Bool;
  default:
    throw std::runtime_error(std::string("PJRT data type: ") +
                             getPJRTBufferTypeString(pjrt_data_type) +
                             " does not have runtime data type equivalent");
  }
}

PJRT_Buffer_Type convertMLIRToPJRTDataType(mlir::Type type) {

  if (mlir::RankedTensorType tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return convertMLIRToPJRTDataType(tensorType.getElementType());
  }

  if (mlir::FloatType floatType =
          mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
    if (floatType.isF64()) {
      return PJRT_Buffer_Type_F64;
    } else if (floatType.isF32()) {
      return PJRT_Buffer_Type_F32;
    } else if (floatType.isF16()) {
      return PJRT_Buffer_Type_F16;
    } else if (floatType.isBF16()) {
      return PJRT_Buffer_Type_BF16;
    } else {
      throw std::runtime_error("Unsupported float type");
    }
  } else if (mlir::IntegerType intType =
                 mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.isSigned() || intType.isSignless()) {
      if (intType.getWidth() == 64) {
        return PJRT_Buffer_Type_S64;
      } else if (intType.getWidth() == 32) {
        return PJRT_Buffer_Type_S32;
      } else if (intType.getWidth() == 16) {
        return PJRT_Buffer_Type_S16;
      } else if (intType.getWidth() == 8) {
        return PJRT_Buffer_Type_S8;
      } else if (intType.getWidth() == 1 && intType.isSignless()) {
        return PJRT_Buffer_Type_PRED; // 1 bit integer is a bool in mlir
      } else {
        throw std::runtime_error("Unsupported signed integer type");
      }
    } else {
      if (intType.getWidth() == 64) {
        return PJRT_Buffer_Type_U64;
      } else if (intType.getWidth() == 32) {
        return PJRT_Buffer_Type_U32;
      } else if (intType.getWidth() == 16) {
        return PJRT_Buffer_Type_U16;
      } else if (intType.getWidth() == 8) {
        return PJRT_Buffer_Type_U8;
      } else if (intType.getWidth() == 1) {
        return PJRT_Buffer_Type_PRED; // 1 bit integer is a bool in mlir
      } else {
        throw std::runtime_error("Unsupported unsigned integer type");
      }
    }
  }

  throw std::runtime_error("Unsupported data type");
}

} // namespace tt::pjrt::data_type_utils

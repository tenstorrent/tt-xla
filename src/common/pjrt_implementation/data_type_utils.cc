// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/pjrt_implementation/data_type_utils.h"

// c++ standard library includes
#include <stdexcept>

namespace tt::pjrt::data_type_utils {

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
  default:
    throw std::runtime_error("Unsupported PJRT buffer data type");
  }
}

} // namespace tt::pjrt::data_type_utils

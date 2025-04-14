// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/utils.h"
#include "common/status.h"

namespace tt::pjrt::utils {

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType) {
  switch (ElementType) {
  case tt::target::DataType::UInt8:
    return PJRT_Buffer_Type_U8;
  case tt::target::DataType::UInt16:
    return PJRT_Buffer_Type_U16;
  case tt::target::DataType::UInt32:
    return PJRT_Buffer_Type_U32;
  case tt::target::DataType::Float16:
    return PJRT_Buffer_Type_F16;
  case tt::target::DataType::Float32:
    return PJRT_Buffer_Type_F32;
  case tt::target::DataType::BFloat16:
    return PJRT_Buffer_Type_BF16;
  case tt::target::DataType::Int32:
    return PJRT_Buffer_Type_S32;
  default:
    assert(false && "Unsupported data type");
    return PJRT_Buffer_Type_BF16;
  }
}

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type) {
  switch (buffer_type) {
  case PJRT_Buffer_Type_U8:
    return std::make_pair(tt::target::DataType::UInt8, 1);
  case PJRT_Buffer_Type_U16:
    return std::make_pair(tt::target::DataType::UInt16, 2);
  case PJRT_Buffer_Type_U32:
    return std::make_pair(tt::target::DataType::UInt32, 4);
  case PJRT_Buffer_Type_F16:
    return std::make_pair(tt::target::DataType::Float16, 2);
  case PJRT_Buffer_Type_F32:
    return std::make_pair(tt::target::DataType::Float32, 4);
  case PJRT_Buffer_Type_BF16:
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  case PJRT_Buffer_Type_S32:
    return std::make_pair(tt::target::DataType::Int32, 4);
  case PJRT_Buffer_Type_INVALID:
  case PJRT_Buffer_Type_S4:
  case PJRT_Buffer_Type_S8:
  case PJRT_Buffer_Type_S16:
  case PJRT_Buffer_Type_S64:
  return std::make_pair(tt::target::DataType::Int32, 4);
  case PJRT_Buffer_Type_U4:
  case PJRT_Buffer_Type_PRED:
  case PJRT_Buffer_Type_U64:
  case PJRT_Buffer_Type_F64:
  case PJRT_Buffer_Type_C64:
  case PJRT_Buffer_Type_C128:
  default:
    auto msg = "Unsupported buffer type: " + std::to_string(buffer_type);
    DLOG_F(LOG_DEBUG, "%s", msg.c_str());
    assert(false && msg.c_str());
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  }
}

} // namespace tt::pjrt::utils

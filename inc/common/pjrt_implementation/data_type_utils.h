// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/types.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

namespace tt::pjrt::data_type_utils {

PJRT_Buffer_Type
convertRuntimeToPJRTDataType(tt::target::DataType runtime_data_type);

tt::target::DataType
convertPJRTToRuntimeDataType(PJRT_Buffer_Type pjrt_data_type);

} // namespace tt::pjrt::data_type_utils

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

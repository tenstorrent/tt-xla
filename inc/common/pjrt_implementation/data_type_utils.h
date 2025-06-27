// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// PJRT C API includes
#include "mlir/IR/Types.h"
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/types.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

namespace tt::pjrt::data_type_utils {

// Returns a the name of the PJRT_Buffer_Type enum as a std::string.
std::string getPJRTBufferTypeString(PJRT_Buffer_Type type);

// Returns the PJRT_Buffer_Type enum corresponding to the given runtime data
// type.
PJRT_Buffer_Type
convertRuntimeToPJRTDataType(tt::target::DataType runtime_data_type);

// Returns the runtime data type corresponding to the given PJRT_Buffer_Type
// enum.
tt::target::DataType
convertPJRTToRuntimeDataType(PJRT_Buffer_Type pjrt_data_type);

// Returns the PJRT_Buffer_Type enum corresponding to the given MLIR type.
PJRT_Buffer_Type convertMLIRToPJRTDataType(mlir::Type type);

} // namespace tt::pjrt::data_type_utils

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

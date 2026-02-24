// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_DATA_TYPE_UTILS_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_DATA_TYPE_UTILS_H_

// llvm mlir includes
#include "mlir/IR/Types.h"

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/types.h"

namespace tt::pjrt::data_type_utils {

// Returns string with a name of the given PJRT buffer data type.
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

// Returns true if the given PJRT buffer type is a complex type (C64 or C128).
// Complex tensors are stored internally as float tensors with a trailing
// dimension of 2 (interleaved real/imag), since the runtime has no complex
// type support.
bool isComplexPJRTType(PJRT_Buffer_Type type);

} // namespace tt::pjrt::data_type_utils

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_DATA_TYPE_UTILS_H_

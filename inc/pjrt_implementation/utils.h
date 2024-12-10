// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <utility>

#include "pjrt_implementation/error_instance.h"
#include "tt/runtime/runtime.h"

#ifndef TT_XLA_UTILS_H_
#define TT_XLA_UTILS_H_

namespace tt::pjrt {

inline PJRT_Error *MakeError(tt_pjrt_status status) {
  if (tt_pjrt_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error *>(alloced_error.release());
}

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType);

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type);

} // namespace tt::pjrt

#endif

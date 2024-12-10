// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "pjrt_implementation/error_instance.h"

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

} // namespace tt::pjrt

#endif

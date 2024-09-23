// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_PJRT_STATUS
#define TT_PJRT_STATUS

#include "loguru/loguru.hpp"

#define LOG_DEBUG 1

namespace tt::pjrt {

  enum class tt_pjrt_status {
    kSuccess = 0,
    kCancelled = 1,
    kUnknown = 2,
    kInvalidArgument = 3,
    kDeadlineExceeded = 4,
    kNotFound = 5,
    kAlreadyExists = 6,
    kPermissionDenied = 7,
    kResourceExhausted = 8,
    kFailedPrecondition = 9,
    kAborted = 10,
    kOutOfRange = 11,
    kUnimplemented = 12,
    kInternal = 13,
    kUnavailable = 14,
    kDataLoss = 15,
    kUnauthenticated = 16,
  };

  inline bool tt_pjrt_status_is_ok(tt_pjrt_status status) {
    return status == tt_pjrt_status::kSuccess;
  }

}
#endif // TT_PJRT_STATUS
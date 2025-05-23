// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_PJRT_STATUS
#define TT_PJRT_STATUS

// TODO(mrakita): Extract logging utils into separate file from status.
#include "loguru/loguru.hpp"

// Loguru doesn't have debug named verbosity and the next verbosity after INFO
// is called `Verbosity_1`, so defining this to avoid doing `DLOG_F(1, ...)`.
#define LOG_DEBUG 1

// TODO(mrakita): Move into `tt::pjrt::status` namespace, rename enum class and
// `tt_pjrt_status_is_ok` function.
namespace tt::pjrt {

// This enum represents the `PJRT_Error_Code` enum with addition of success
// code. Codes are based on https://abseil.io/docs/cpp/guides/status-codes.
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

} // namespace tt::pjrt

#endif // TT_PJRT_STATUS

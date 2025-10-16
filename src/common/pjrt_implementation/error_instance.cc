// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/error_instance.h"

// tracy includes
#include <tracy/Tracy.hpp>

namespace tt::pjrt {

std::unique_ptr<ErrorInstance> ErrorInstance::makeError(tt_pjrt_status status) {
  if (tt_pjrt_status_is_ok(status)) {
    return nullptr;
  }

  struct make_unique_enabler : public ErrorInstance {
    make_unique_enabler(tt_pjrt_status status) : ErrorInstance(status) {}
  };

  return std::make_unique<make_unique_enabler>(status);
}

ErrorInstance::ErrorInstance(tt_pjrt_status status)
    : m_status(status),
      m_message(
          tt_pjrt_status_is_ok(status)
              ? "OK"
              : ("Error code: " + std::to_string(static_cast<int>(status)))) {}

void ErrorInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Error_Destroy = internal::onErrorDestroy;
  api->PJRT_Error_Message = internal::onErrorMessage;
  api->PJRT_Error_GetCode = internal::onErrorGetCode;
}

namespace internal {

void onErrorDestroy(PJRT_Error_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "ErrorInstance::PJRT_Error_Destroy");

  delete ErrorInstance::unwrap(args->error);
}

void onErrorMessage(PJRT_Error_Message_Args *args) {
  DLOG_F(LOG_DEBUG, "ErrorInstance::PJRT_Error_Message");

  const std::string &message = ErrorInstance::unwrap(args->error)->getMessage();

  args->message = message.data();
  args->message_size = message.size();
}

PJRT_Error *onErrorGetCode(PJRT_Error_GetCode_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ErrorInstance::PJRT_Error_GetCode");

  tt_pjrt_status status = ErrorInstance::unwrap(args->error)->getStatus();

  switch (status) {
  case tt_pjrt_status::kCancelled:
    args->code = PJRT_Error_Code_CANCELLED;
    break;
  case tt_pjrt_status::kUnknown:
    args->code = PJRT_Error_Code_UNKNOWN;
    break;
  case tt_pjrt_status::kInvalidArgument:
    args->code = PJRT_Error_Code_INVALID_ARGUMENT;
    break;
  case tt_pjrt_status::kDeadlineExceeded:
    args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
    break;
  case tt_pjrt_status::kNotFound:
    args->code = PJRT_Error_Code_NOT_FOUND;
    break;
  case tt_pjrt_status::kAlreadyExists:
    args->code = PJRT_Error_Code_ALREADY_EXISTS;
    break;
  case tt_pjrt_status::kPermissionDenied:
    args->code = PJRT_Error_Code_PERMISSION_DENIED;
    break;
  case tt_pjrt_status::kResourceExhausted:
    args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
    break;
  case tt_pjrt_status::kFailedPrecondition:
    args->code = PJRT_Error_Code_FAILED_PRECONDITION;
    break;
  case tt_pjrt_status::kAborted:
    args->code = PJRT_Error_Code_ABORTED;
    break;
  case tt_pjrt_status::kOutOfRange:
    args->code = PJRT_Error_Code_OUT_OF_RANGE;
    break;
  case tt_pjrt_status::kUnimplemented:
    args->code = PJRT_Error_Code_UNIMPLEMENTED;
    break;
  case tt_pjrt_status::kInternal:
    args->code = PJRT_Error_Code_INTERNAL;
    break;
  case tt_pjrt_status::kUnavailable:
    args->code = PJRT_Error_Code_UNAVAILABLE;
    break;
  case tt_pjrt_status::kDataLoss:
    args->code = PJRT_Error_Code_DATA_LOSS;
    break;
  case tt_pjrt_status::kUnauthenticated:
    args->code = PJRT_Error_Code_UNAUTHENTICATED;
    break;
  default:
    // Should not happen.
    DLOG_F(WARNING, "Encountered unknown PJRT status code: %d",
           static_cast<int>(status));

    args->code = PJRT_Error_Code_UNKNOWN;
  }

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

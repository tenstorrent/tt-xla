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

#include <memory>
#include <string>

namespace tt::pjrt {

void ErrorInstance::BindApi(PJRT_Api *api) {
  api->PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args *args) {
    if (!args->error)
      return;
    delete ErrorInstance::FromError(args->error);
  };
  api->PJRT_Error_Message = +[](PJRT_Error_Message_Args *args) {
    const ErrorInstance *error = ErrorInstance::FromError(args->error);
    if (!error) {
      args->message = "OK";
      args->message_size = 2;
      return;
    }

    const std::string &message = error->message();
    args->message = message.data();
    args->message_size = message.size();
  };
  api->PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args *args) -> PJRT_Error * {
    const ErrorInstance *error = ErrorInstance::FromError(args->error);
    tt_pjrt_status status = error->status();
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
      args->code = PJRT_Error_Code_UNKNOWN;
    }
    return nullptr;
  };
}

const std::string &ErrorInstance::message() const {
  std::string buffer;
  buffer.reserve(256);
  buffer += "Error code: ";
  buffer += std::to_string(static_cast<int>(status_));
  cached_message_ = std::move(buffer);
  return cached_message_;
}

PJRT_Error *ErrorInstance::MakeError(tt_pjrt_status status) {
  if (tt_pjrt_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error *>(alloced_error.release());
}

} // namespace tt::pjrt

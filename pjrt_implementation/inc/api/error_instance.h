// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <string>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "utils/status.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_ERROR_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_ERROR_INSTANCE_H_

namespace tt::pjrt {

// Represents PJRT_Error structure and the functionality around it.
class ErrorInstance {

public:
  // Creates new error instance.
  static std::unique_ptr<ErrorInstance> makeError(tt_pjrt_status status);

  // Binds PJRT API functions implementation related to PJRT_Error structure.
  static void bindApi(PJRT_Api *api);

  // Casts this error instance to PJRT_Error pointer.
  operator PJRT_Error *() { return reinterpret_cast<PJRT_Error *>(this); }

  // Casts the PJRT_Error pointer to ErrorInstance pointer.
  static const ErrorInstance *unwrap(const PJRT_Error *error) {
    return reinterpret_cast<const ErrorInstance *>(error);
  }

  // Returns error status.
  tt_pjrt_status getStatus() const { return m_status; }

  // Returns error message.
  const std::string &getMessage() const { return m_message; }

private:
  // Constructs error instance from the given status.
  ErrorInstance(tt_pjrt_status status);

  // Error status.
  const tt_pjrt_status m_status;

  // Error message.
  const std::string m_message;
};

namespace internal {

// Implements PJRT_Error_Destroy API function.
void onErrorDestroy(PJRT_Error_Destroy_Args *args);

// Implements PJRT_Error_Message API function.
void onErrorMessage(PJRT_Error_Message_Args *args);

// Implements PJRT_Error_GetCode API function.
PJRT_Error *onErrorGetCode(PJRT_Error_GetCode_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_ERROR_INSTANCE_H_

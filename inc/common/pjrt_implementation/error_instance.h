// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/status.h"


#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_ERROR_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_ERROR_INSTANCE_H_

namespace tt::pjrt {

class ErrorInstance {

public:
  ErrorInstance(tt_pjrt_status status) : status_(status) {}
  ~ErrorInstance() {}
  static void BindApi(PJRT_Api *api);

  static const ErrorInstance *FromError(const PJRT_Error *error) {
    return reinterpret_cast<const ErrorInstance *>(error);
  }

  tt_pjrt_status status() const { return status_; }
  const std::string &message() const;

  static PJRT_Error *MakeError(tt_pjrt_status status);

private:
  tt_pjrt_status status_;
  mutable std::string cached_message_;
};

} // namespace tt::pjrt

#endif

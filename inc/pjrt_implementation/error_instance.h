// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "common/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_ERROR_INSTANCE_H_
#define TT_XLA_ERROR_INSTANCE_H_

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

private:
  tt_pjrt_status status_;
  mutable std::string cached_message_;
};

} // namespace tt::pjrt

#endif

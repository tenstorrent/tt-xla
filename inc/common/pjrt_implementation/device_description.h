// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include <sstream>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_

namespace tt::pjrt {

class DeviceDescription {

public:
  DeviceDescription(int32_t client_id) : client_id_(client_id) {};
  ~DeviceDescription();
  operator PJRT_DeviceDescription *() {
    return reinterpret_cast<PJRT_DeviceDescription *>(this);
  }
  static void BindApi(PJRT_Api *api);

  static DeviceDescription *Unwrap(PJRT_DeviceDescription *device) {
    return reinterpret_cast<DeviceDescription *>(device);
  }

  std::string_view kind_string() { return kind_string_; }
  std::string_view debug_string() { return to_string(); }
  std::string_view to_string() {
    std::stringstream ss;
    ss << kind_string_ << "(id=" << device_id() << ", arch=" << arch_string_
       << ")";
    user_string_ = ss.str();
    return user_string_;
  }

  // TODO
  int64_t device_id() { return 0; }

  int client_id() { return client_id_; }

  int process_index() { return 0; }

private:
  int client_id_;
  
  // TODO We should understand better how these are used.
  // See https://github.com/tenstorrent/tt-xla/issues/125
  std::string kind_string_ = "TTDevice";
  std::string arch_string_ = "Wormhole";
  std::string user_string_ = "";
};

} // namespace tt::pjrt

#endif

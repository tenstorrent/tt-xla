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

#include "types_generated.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_

namespace tt::pjrt {

class DeviceDescription {

public:
  DeviceDescription(int32_t client_id, tt::target::Arch arch)
      : client_id_(client_id) {
    kind_string_ = target::EnumNameArch(arch);
    std::stringstream ss;
    ss << "TTDevice(id=" << device_id() << ", arch=" << kind_string_ << ")";
    user_string_ = ss.str();
  };
  ~DeviceDescription();
  operator PJRT_DeviceDescription *() {
    return reinterpret_cast<PJRT_DeviceDescription *>(this);
  }
  static void BindApi(PJRT_Api *api);

  static DeviceDescription *Unwrap(PJRT_DeviceDescription *device) {
    return reinterpret_cast<DeviceDescription *>(device);
  }

  std::string_view kind_string() { return kind_string_; }
  std::string_view debug_string() { return user_string_; }
  std::string_view to_string() { return user_string_; }

  // TODO
  int64_t device_id() { return 0; }

  int client_id() { return client_id_; }

  int process_index() { return 0; }

private:
  int client_id_;

  std::string kind_string_;
  std::string user_string_;
};

} // namespace tt::pjrt

#endif

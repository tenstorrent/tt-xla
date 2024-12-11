// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_DEVICE_DESCRIPTION_H_
#define TT_XLA_DEVICE_DESCRIPTION_H_

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
  std::string_view debug_string() { return debug_string_; }
  std::string_view user_string() {
    std::stringstream ss;
    ss << "TTDevice(id=" << device_id() << ")";
    user_string_ = ss.str();
    return user_string_;
  }
  // TODO
  int64_t device_id() { return 0; }

  int client_id() { return client_id_; }

  int process_index() { return 0; }

private:
  int client_id_;
  std::string kind_string_ = "wormhole";
  std::string debug_string_ = "debug_string";
  std::string user_string_ = "";
};

} // namespace tt::pjrt

#endif

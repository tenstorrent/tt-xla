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

// tt-mlir includes
#include "types_generated.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_DESCRIPTION_H_

namespace tt::pjrt {

class DeviceDescription {

public:
  DeviceDescription(int32_t device_id, tt::target::Arch arch);
  ~DeviceDescription();
  operator PJRT_DeviceDescription *() {
    return reinterpret_cast<PJRT_DeviceDescription *>(this);
  }
  static void BindApi(PJRT_Api *api);

  static DeviceDescription *Unwrap(PJRT_DeviceDescription *device) {
    return reinterpret_cast<DeviceDescription *>(device);
  }

  // Returns a vendor-dependent string that uniquely identifies the kind of
  // device, e.g. `Wormhole_b0`.
  const std::string &getDeviceKind() const { return m_device_kind; }

  // Returns a debug string suitable for logging when errors occur. Should be
  // verbose enough to describe the current device unambiguously.
  const std::string &toDebugString() const { return m_user_string; }

  // Returns a device description string suitable for reading by end users,
  // should be reasonably terse.
  const std::string &toString() const { return m_user_string; }

  int getDeviceId() const { return m_device_id; }

  int getProcessIndex() const { return 0; }

private:
  int m_device_id;

  std::string m_device_kind;

  std::string m_user_string;
};

} // namespace tt::pjrt

#endif

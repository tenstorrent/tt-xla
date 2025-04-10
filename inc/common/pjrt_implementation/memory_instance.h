// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt


#include "xla/pjrt/c/pjrt_c_api.h"
#include <vector>

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance;

class MemoryInstance {

public:
    MemoryInstance(std::vector<DeviceInstance *> addressable_by_devices) : addressable_by_devices_(addressable_by_devices) {}

    MemoryInstance() = default;
    ~MemoryInstance();

  operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }
  static void BindApi(PJRT_Api *api);
  static MemoryInstance *Unwrap(PJRT_Memory *device_description) {
    return reinterpret_cast<MemoryInstance *>(device_description);
  }

  void addDevice(DeviceInstance *device) { addressable_by_devices_.push_back(device); }

  const std::vector<DeviceInstance *> &addressable_by_devices() { return addressable_by_devices_; }

private:
    std::vector<DeviceInstance *> addressable_by_devices_;
};

} // namespace tt::pjrt

#endif
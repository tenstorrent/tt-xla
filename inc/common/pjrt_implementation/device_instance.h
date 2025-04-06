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

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "common/pjrt_implementation/device_description.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance {

public:
  // Creates new device instance.
  static std::unique_ptr<DeviceInstance> createInstance(int global_device_id,
                                                        bool is_addressable,
                                                        int local_device_id,
                                                        tt::target::Arch arch);

  // Binds PJRT API functions implementation related to PJRT_Device structure.
  static void bindApi(PJRT_Api *api);

  // Casts this device instance to PJRT_Device and returns pointer to it.
  operator PJRT_Device *() { return reinterpret_cast<PJRT_Device *>(this); }

  // Casts the PJRT_Device pointer to DeviceInstance pointer.
  static DeviceInstance *unwrap(PJRT_Device *device) {
    return reinterpret_cast<DeviceInstance *>(device);
  }

  // Returns reference to device description.
  DeviceDescription &getDeviceDescription() { return m_description; }

  // Returns const reference to device description.
  const DeviceDescription &getDeviceDescription() const {
    return m_description;
  }

  // Returns true if device is addressable.
  bool isAddressable() const { return m_is_addressable; }

  // Returns local device ID.
  int getLocalDeviceId() const { return m_local_device_id; }

  // Returns global device ID.
  int getGlobalDeviceId() const { return m_description.getDeviceId(); }

private:
  // Constructor.
  DeviceInstance(int global_device_id, bool is_addressable, int local_device_id,
                 tt::target::Arch arch)
      : m_description(global_device_id, arch), m_is_addressable(is_addressable),
        m_local_device_id(local_device_id) {}

  // Device description.
  DeviceDescription m_description;

  // True if device is addressable. Addressable devices are those that the
  // client can issue commands to.
  bool m_is_addressable;

  // Local ID of this device unique between all addressable devices.
  int m_local_device_id;
};

namespace internal {

// Implements PJRT_Device_GetDescription API function.
PJRT_Error *onDeviceGetDescription(PJRT_Device_GetDescription_Args *args);

// Implements PJRT_Device_IsAddressable API function.
PJRT_Error *onDeviceIsAddressable(PJRT_Device_IsAddressable_Args *args);

// Implements PJRT_Device_LocalHardwareId API function.
PJRT_Error *onDeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

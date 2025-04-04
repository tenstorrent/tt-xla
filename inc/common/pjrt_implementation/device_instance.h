// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/pjrt_implementation/device_description.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance {

public:
  // Constructor.
  DeviceInstance(int global_device_id, int local_device_id, bool is_addressable,
                 tt::target::Arch arch)
      : m_description(global_device_id, arch), m_is_addressable(is_addressable),
        m_local_device_id(local_device_id) {}

  // Binds PJRT API functions implementation related to PJRT_Device structure.
  static void bindApi(PJRT_Api *api);

  // Casts this device instance to PJRT_Device and returns pointer to it.
  operator PJRT_Device *() { return reinterpret_cast<PJRT_Device *>(this); }

  // Casts the PJRT_Device pointer to DeviceInstance pointer.
  static DeviceInstance *unwrap(PJRT_Device *device) {
    return reinterpret_cast<DeviceInstance *>(device);
  }

  // Returns pointer to device description.
  DeviceDescription *getDeviceDescription() { return &m_description; }

  // Returns const reference to device description.
  const DeviceDescription &getDeviceDescription() const {
    return m_description;
  }

  // Returns true if device is addressable.
  bool isAddressable() const { return m_is_addressable; }

  // Returns local device ID.
  int getLocalDeviceId() const { return m_local_device_id; }

  tt_pjrt_status
  HostBufferToDevice(const void *data, PJRT_Buffer_Type type,
                     const int64_t *dims, size_t num_dims,
                     const int64_t *byte_strides, size_t num_byte_strides,
                     PJRT_HostBufferSemantics host_buffer_semantics,
                     EventInstance **out_done_with_host_buffer_event,
                     BufferInstance **out_buffer);

private:
  static size_t getTensorSize(const std::vector<std::uint32_t> &shape,
                              size_t element_size);

  // Create a buffer instance from a host data pointer, by copying it into
  // another memory. This is necessary as we have no ownership of the passed
  // pointer, and it might happen that the pointer is deallocated before the
  // buffer is used. See issue #248 for more details.
  std::unique_ptr<BufferInstance>
  MakeDeviceBuffer(const void *data_ptr, std::vector<std::uint32_t> &shape,
                   std::vector<std::uint32_t> &strides, size_t element_size,
                   tt::target::DataType element_type);

  // Device description.
  DeviceDescription m_description;

  // True if device is addressable. Addressable devices are those that the
  // client can issue commands to.
  bool m_is_addressable;

  // Local ID of this device unique between all addressable devices.
  int m_local_device_id;
};

namespace internal {

PJRT_Error *onDeviceGetDescription(PJRT_Device_GetDescription_Args *args);

PJRT_Error *onDeviceIsAddressable(PJRT_Device_IsAddressable_Args *args);

PJRT_Error *onDeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

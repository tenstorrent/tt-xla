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
#include "common/pjrt_implementation/event_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

namespace tt::pjrt {

class ClientInstance;
class BufferInstance;

class DeviceInstance {

public:
  DeviceInstance(int client_id, ClientInstance &client, tt::target::Arch arch)
      : client_(client), description_(client_id, arch) {}
  ~DeviceInstance();
  operator PJRT_Device *() { return reinterpret_cast<PJRT_Device *>(this); }
  static void BindApi(PJRT_Api *api);

  static DeviceInstance *Unwrap(PJRT_Device *device) {
    return reinterpret_cast<DeviceInstance *>(device);
  }

  static DeviceInstance *Unwrap(PJRT_DeviceDescription *device_description) {
    return reinterpret_cast<DeviceInstance *>(device_description);
  }
  ClientInstance &client() { return client_; }
  bool is_addressable() { return true; }
  int local_hardware_id() { return -1; }

  tt_pjrt_status
  HostBufferToDeviceZeroDim(PJRT_Buffer_Type type, const int64_t *dims,
                            size_t num_dims,
                            EventInstance **out_done_with_host_buffer_event,
                            BufferInstance **out_buffer);

  tt_pjrt_status
  HostBufferToDeviceSplat(const void *data, PJRT_Buffer_Type type,
                          const int64_t *dims, size_t num_dims,
                          EventInstance **out_done_with_host_buffer_event,
                          BufferInstance **out_buffer);

  tt_pjrt_status
  HostBufferToDevice(const void *data, PJRT_Buffer_Type type,
                     const int64_t *dims, size_t num_dims,
                     const int64_t *byte_strides, size_t num_byte_strides,
                     PJRT_HostBufferSemantics host_buffer_semantics,
                     EventInstance **out_done_with_host_buffer_event,
                     BufferInstance **out_buffer);

  DeviceDescription *device_description() { return &description_; }

private:
  tt_pjrt_status OpenDevice();

  ClientInstance &client_;
  uint64_t last_transfer_timepoint_ = 0;
  DeviceDescription description_;
};

} // namespace tt::pjrt

#endif

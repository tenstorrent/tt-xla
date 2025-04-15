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

#include "tt/runtime/runtime.h"

#include "common/pjrt_implementation/device_description.h"
#include "common/pjrt_implementation/event_instance.h"
#include "common/pjrt_implementation/memory_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_DEVICE_INSTANCE_H_

namespace tt::pjrt {

class ClientInstance;
class BufferInstance;

class DeviceInstance {

public:
  DeviceInstance(int device_id, ClientInstance &client, tt::target::Arch arch, std::vector<MemoryInstance *> addressable_memories)
      : client_(client), description_(device_id, arch), addressable_memories_(addressable_memories) {
        for (auto memory : addressable_memories_) {
          memory->addDevice(this);
          DLOG_F(LOG_DEBUG, "DeviceInstance ADDRT %p of memory %p", this, memory);
      }
  }
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
  const std::vector<MemoryInstance *> &addressable_memories() { return addressable_memories_; }
  MemoryInstance *default_memory() { return addressable_memories_[0]; }
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

  DeviceDescription *device_description() { return &description_; }
  const DeviceDescription *device_description() const { return &description_; }

private:
  tt_pjrt_status OpenDevice();

  ClientInstance &client_;
  uint64_t last_transfer_timepoint_ = 0;
  DeviceDescription description_;
  std::vector<MemoryInstance *> addressable_memories_;
};

size_t getTensorSize(const std::vector<std::uint32_t> &shape,
  size_t element_size);

// Create a buffer instance from a host data pointer, by copying it into
// another memory. This is necessary as we have no ownership of the passed
// pointer, and it might happen that the pointer is deallocated before the
// buffer is used. See issue #248 for more details.
std::unique_ptr<BufferInstance>
MakeDeviceBuffer(DeviceInstance *device, const void *data_ptr, std::vector<std::uint32_t> &shape,
                  std::vector<std::uint32_t> &strides, size_t element_size,
                  tt::target::DataType element_type);

tt_pjrt_status
HostBufferToDevice(DeviceInstance *device, const void *data, PJRT_Buffer_Type type,
                    const int64_t *dims, size_t num_dims,
                    const int64_t *byte_strides, size_t num_byte_strides,
                    PJRT_HostBufferSemantics host_buffer_semantics,
                    EventInstance **out_done_with_host_buffer_event,
                    BufferInstance **out_buffer);

} // namespace tt::pjrt

#endif

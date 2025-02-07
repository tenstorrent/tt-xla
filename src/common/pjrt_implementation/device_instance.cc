// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/device_instance.h"

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/utils.h"
#include "common/status.h"

namespace tt::pjrt {

DeviceInstance::~DeviceInstance() = default;

void DeviceInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");
  api->PJRT_Device_IsAddressable =
      +[](PJRT_Device_IsAddressable_Args *args) -> PJRT_Error * {
    args->is_addressable =
        DeviceInstance::Unwrap(args->device)->is_addressable();
    return nullptr;
  };
  api->PJRT_Device_LocalHardwareId =
      +[](PJRT_Device_LocalHardwareId_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_LocalHardwareId_Args");
    args->local_hardware_id =
        DeviceInstance::Unwrap(args->device)->local_hardware_id();
    return nullptr;
  };
  api->PJRT_Device_AddressableMemories =
      +[](PJRT_Device_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_AddressableMemories");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Device_DefaultMemory =
      +[](PJRT_Device_DefaultMemory_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_DefaultMemory");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Device_GetDescription =
      +[](PJRT_Device_GetDescription_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetDescription");
    args->device_description = reinterpret_cast<PJRT_DeviceDescription *>(
        DeviceInstance::Unwrap(args->device)->device_description());
    return nullptr;
  };
}

tt_pjrt_status DeviceInstance::OpenDevice() {
  DLOG_F(LOG_DEBUG, "DeviceInstance::OpenDevice");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status DeviceInstance::HostBufferToDevice(
    const void *data, PJRT_Buffer_Type type, const int64_t *dims,
    size_t num_dims, const int64_t *byte_strides, size_t num_byte_strides,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance **out_done_with_host_buffer_event,
    BufferInstance **out_buffer) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::HostBufferToDevice");

  std::pair<tt::target::DataType, size_t> tt_buffer_type =
      tt::pjrt::utils::MapBufferTypeToElementType(type);
  tt::target::DataType element_type = tt_buffer_type.first;
  size_t element_size = tt_buffer_type.second;
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> strides;
  if (num_dims == 0) {
    shape.push_back(1);
    strides.push_back(1);
  }
  assert(num_byte_strides == num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    shape.push_back(dims[i]);
    strides.push_back(byte_strides[i] / element_size);
  }
  BufferInstance *buffer_instance =
      MakeDeviceBuffer(data, shape, strides, element_size, element_type);
  DLOG_F(INFO, "Buffer created with id: %d", buffer_instance->unique_id());
  buffer_instance->setType(type);
  *out_buffer = buffer_instance;
  EventInstance *event_instance = new EventInstance();
  *out_done_with_host_buffer_event = event_instance;
  return tt_pjrt_status::kSuccess;
}

size_t DeviceInstance::getTensorSize(const std::vector<std::uint32_t> &shape,
                               size_t element_size) {
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size * element_size;
}

BufferInstance *DeviceInstance::MakeDeviceBuffer(
    const void *data, std::vector<std::uint32_t> &shape,
    std::vector<std::uint32_t> &strides, size_t element_size,
    tt::target::DataType element_type) {
  size_t tensor_size = getTensorSize(shape, element_size);
  std::shared_ptr<void> new_memory(new char[tensor_size], [](void *ptr) {
    delete[] static_cast<char *>(ptr);
  });
  std::memcpy(new_memory.get(), data, tensor_size);
  std::unique_ptr<tt::runtime::Tensor> device_tensor =
      std::make_unique<tt::runtime::Tensor>(tt::runtime::createTensor(
          new_memory, shape, strides, element_size, element_type));
  return new BufferInstance(*this, device_tensor, shape, strides, new_memory);
}

} // namespace tt::pjrt

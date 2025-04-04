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

namespace tt::pjrt {

void DeviceInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");
  api->PJRT_Device_GetDescription = internal::onDeviceGetDescription;
  api->PJRT_Device_IsAddressable = internal::onDeviceIsAddressable;
  api->PJRT_Device_LocalHardwareId = internal::onDeviceLocalHardwareId;
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
  std::unique_ptr<BufferInstance> buffer_instance =
      MakeDeviceBuffer(data, shape, strides, element_size, element_type);
  buffer_instance->setType(type);
  *out_buffer = buffer_instance.release();
  EventInstance *event_instance = new EventInstance();
  *out_done_with_host_buffer_event = event_instance;
  return tt_pjrt_status::kSuccess;
}

size_t DeviceInstance::getTensorSize(const std::vector<std::uint32_t> &shape,
                                     size_t element_size) {
  std::uint32_t elementsCount = std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<std::uint32_t>());

  return static_cast<size_t>(elementsCount) * element_size;
}

std::unique_ptr<BufferInstance> DeviceInstance::MakeDeviceBuffer(
    const void *data, std::vector<std::uint32_t> &shape,
    std::vector<std::uint32_t> &strides, size_t element_size,
    tt::target::DataType element_type) {
  size_t tensor_size = getTensorSize(shape, element_size);

  // TODO_OOM: No need for this since createOwnedTensor creates its own buffer
  // but we need this until runtime API changes in order to have a pointer to
  // host memory to create multi device tensor from.
  std::shared_ptr<void> new_memory(new std::byte[tensor_size], [](void *ptr) {
    delete[] static_cast<std::byte *>(ptr);
  });

  std::memcpy(new_memory.get(), data, tensor_size);

  tt::runtime::Tensor device_tensor = tt::runtime::createOwnedTensor(
      new_memory, shape, strides, element_size, element_type);

  std::pair<tt::target::DataType, size_t> tt_buffer_type = {element_type,
                                                            element_size};

  return std::make_unique<BufferInstance>(*this, device_tensor, shape, strides,
                                          tt_buffer_type, new_memory);
}

namespace internal {

PJRT_Error *onDeviceGetDescription(PJRT_Device_GetDescription_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetDescription");

  args->device_description =
      *DeviceInstance::unwrap(args->device)->getDeviceDescription();

  return nullptr;
}

PJRT_Error *onDeviceIsAddressable(PJRT_Device_IsAddressable_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_IsAddressable");

  args->is_addressable = DeviceInstance::unwrap(args->device)->isAddressable();

  return nullptr;
}

PJRT_Error *onDeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_LocalHardwareId");

  args->local_hardware_id =
      DeviceInstance::unwrap(args->device)->getLocalDeviceId();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

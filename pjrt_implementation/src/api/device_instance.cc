// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/device_instance.h"

// tt-xla includes
#include "api/memory_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {

std::unique_ptr<DeviceInstance>
DeviceInstance::createInstance(int global_device_id, bool is_addressable,
                               int local_device_id, tt::target::Arch arch) {
  struct make_unique_enabler : public DeviceInstance {
    make_unique_enabler(int global_device_id, bool is_addressable,
                        int local_device_id, tt::target::Arch arch)
        : DeviceInstance(global_device_id, is_addressable, local_device_id,
                         arch) {}
  };

  return std::make_unique<make_unique_enabler>(global_device_id, is_addressable,
                                               local_device_id, arch);
}

void DeviceInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Device_GetDescription = internal::onDeviceGetDescription;
  api->PJRT_Device_IsAddressable = internal::onDeviceIsAddressable;
  api->PJRT_Device_LocalHardwareId = internal::onDeviceLocalHardwareId;
  api->PJRT_Device_AddressableMemories = internal::onDeviceAddressableMemories;
  api->PJRT_Device_DefaultMemory = internal::onDeviceDefaultMemory;
}

namespace internal {

PJRT_Error *onDeviceGetDescription(PJRT_Device_GetDescription_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetDescription");

  args->device_description =
      DeviceInstance::unwrap(args->device)->getDeviceDescription();

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

PJRT_Error *
onDeviceAddressableMemories(PJRT_Device_AddressableMemories_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_AddressableMemories");

  DeviceInstance *device = DeviceInstance::unwrap(args->device);
  args->memories = reinterpret_cast<PJRT_Memory *const *>(
      device->getAddressableMemories().data());
  args->num_memories = device->getAddressableMemories().size();

  return nullptr;
};

PJRT_Error *onDeviceDefaultMemory(PJRT_Device_DefaultMemory_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_DefaultMemory");

  args->memory = *(DeviceInstance::unwrap(args->device)->getDefaultMemory());

  return nullptr;
};

} // namespace internal

} // namespace tt::pjrt

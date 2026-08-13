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

// tracy includes
#include "tracy/Tracy.hpp"

// tt-xla includes
#include "api/memory_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {

std::unique_ptr<DeviceInstance> DeviceInstance::createInstance(
    ClientInstance *client, int global_device_id, bool is_addressable,
    int local_device_id, tt::target::Arch arch, uint64_t dram_size_bytes) {
  struct make_unique_enabler : public DeviceInstance {
    make_unique_enabler(ClientInstance *client, int global_device_id,
                        bool is_addressable, int local_device_id,
                        tt::target::Arch arch, uint64_t dram_size_bytes)
        : DeviceInstance(client, global_device_id, is_addressable,
                         local_device_id, arch, dram_size_bytes) {}
  };

  return std::make_unique<make_unique_enabler>(client, global_device_id,
                                               is_addressable, local_device_id,
                                               arch, dram_size_bytes);
}

void DeviceInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Device_GetDescription = internal::onDeviceGetDescription;
  api->PJRT_Device_IsAddressable = internal::onDeviceIsAddressable;
  api->PJRT_Device_LocalHardwareId = internal::onDeviceLocalHardwareId;
  api->PJRT_Device_AddressableMemories = internal::onDeviceAddressableMemories;
  api->PJRT_Device_DefaultMemory = internal::onDeviceDefaultMemory;
  api->PJRT_Device_GetAttributes = internal::onDeviceGetAttributes;
}

namespace internal {

PJRT_Error *onDeviceGetDescription(PJRT_Device_GetDescription_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetDescription");

  args->device_description =
      DeviceInstance::unwrap(args->device)->getDeviceDescription();

  return nullptr;
}

PJRT_Error *onDeviceIsAddressable(PJRT_Device_IsAddressable_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_IsAddressable");

  args->is_addressable = DeviceInstance::unwrap(args->device)->isAddressable();

  return nullptr;
}

PJRT_Error *onDeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_LocalHardwareId");

  args->local_hardware_id =
      DeviceInstance::unwrap(args->device)->getLocalDeviceId();

  return nullptr;
}

PJRT_Error *
onDeviceAddressableMemories(PJRT_Device_AddressableMemories_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_AddressableMemories");

  DeviceInstance *device = DeviceInstance::unwrap(args->device);
  args->memories = reinterpret_cast<PJRT_Memory *const *>(
      device->getAddressableMemories().data());
  args->num_memories = device->getAddressableMemories().size();

  return nullptr;
};

PJRT_Error *onDeviceDefaultMemory(PJRT_Device_DefaultMemory_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_DefaultMemory");

  args->memory = *(DeviceInstance::unwrap(args->device)->getDefaultMemory());

  return nullptr;
};

PJRT_Error *onDeviceGetAttributes(PJRT_Device_GetAttributes_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetAttributes");

  // Expose the device-description attributes (e.g. "device_arch") at the device
  // level as well. Frameworks read per-device metadata through this device-level
  // call -- torch_xla's global_runtime_device_attributes and jaxlib's
  // PjRtCApiDevice both populate from PJRT_Device_GetAttributes, not from the
  // device description -- so reporting an empty set here hides "device_arch".
  const std::vector<PJRT_NamedValue> &attributes =
      DeviceInstance::unwrap(args->device)->getDeviceDescription().getAttributes();
  args->attributes = attributes.data();
  args->num_attributes = attributes.size();

  // `attributes` points at storage owned by the DeviceDescription, which
  // outlives this call, so there is no backing buffer to free: hand back a null
  // `device_attributes` and a no-op deleter. The deleter must be non-null
  // (jaxlib CHECK-fails otherwise) and the caller invokes it after copying the
  // values.
  args->device_attributes = nullptr;
  args->attributes_deleter = +[](PJRT_Device_Attributes *) {};

  return nullptr;
};

} // namespace internal

} // namespace tt::pjrt

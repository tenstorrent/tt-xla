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

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/client_instance.h"
#include "api/error_instance.h"
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
  api->PJRT_Device_MemoryStats = internal::onDeviceMemoryStats;
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

PJRT_Error *onDeviceMemoryStats(PJRT_Device_MemoryStats_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_MemoryStats");

  const std::optional<tt::runtime::Device> &mesh =
      DeviceInstance::unwrap(args->device)->getClient()->parentMesh();
  if (!mesh.has_value()) {
    return *ErrorInstance::makeError(tt_pjrt_status::kUnavailable).release();
  }

  std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
      memory_view = tt::runtime::getMemoryView(*mesh);
  auto dram = memory_view.find(tt::runtime::MemoryBufferType::DRAM);
  if (dram == memory_view.end()) {
    return *ErrorInstance::makeError(tt_pjrt_status::kUnavailable).release();
  }

  // The runtime reports per-bank figures; scale by the bank count to get the
  // per-device DRAM totals that PJRT clients expect.
  const tt::runtime::MemoryView &view = dram->second;
  args->bytes_in_use =
      static_cast<int64_t>(view.totalBytesAllocatedPerBank * view.numBanks);
  args->bytes_limit =
      static_cast<int64_t>(view.totalBytesPerBank * view.numBanks);
  args->bytes_limit_is_set = true;
  args->largest_free_block_bytes = static_cast<int64_t>(
      view.largestContiguousBytesFreePerBank * view.numBanks);
  args->largest_free_block_bytes_is_set = true;

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

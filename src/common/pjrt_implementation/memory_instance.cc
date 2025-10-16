// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/memory_instance.h"

// c++ standard library includes
#include <cassert>
#include <cstring>

// tracy includes
#include <tracy/Tracy.hpp>

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt {

const std::string MemoryInstance::c_host_memory_kind_name = "pinned_host";
const std::string MemoryInstance::c_device_memory_kind_name = "device";

std::unique_ptr<MemoryInstance> MemoryInstance::createInstance(
    std::vector<DeviceInstance *> &addressable_by_devices, size_t id,
    bool is_host_memory) {
  struct make_unique_enabler : public MemoryInstance {
    make_unique_enabler(std::vector<DeviceInstance *> &addressable_by_devices,
                        size_t id, bool is_host_memory)
        : MemoryInstance(addressable_by_devices, id, is_host_memory) {}
  };
  return std::make_unique<make_unique_enabler>(addressable_by_devices, id,
                                               is_host_memory);
}

void MemoryInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Memory_AddressableByDevices =
      internal::onMemoryAddressableByDevices;
  api->PJRT_Memory_Kind = internal::onMemoryKind;
  api->PJRT_Memory_Id = internal::onMemoryId;
  api->PJRT_Memory_DebugString = internal::onMemoryDebugString;
  api->PJRT_Memory_ToString = internal::onMemoryToString;
  api->PJRT_Memory_Kind_Id = internal::onMemoryKindId;
}

MemoryInstance::MemoryInstance(
    std::vector<DeviceInstance *> &addressable_by_devices, size_t id,
    bool is_host_memory)
    : m_addressable_by_devices(addressable_by_devices), m_id(id),
      m_is_host_memory(is_host_memory) {
  m_debug_string =
      "MemoryInstance: " + std::to_string(id) + " (" + getMemoryKind() + ")";
}

DeviceInstance *MemoryInstance::getDevice() {
  DLOG_F(LOG_DEBUG, "MemoryInstance::getDevice");

  if (isHostMemory()) {
    DLOG_F(WARNING,
           "MemoryInstance::getDevice: Host memory does not have a device.");

    return nullptr;
  }

  assert(m_addressable_by_devices.size() == 1 &&
         "MemoryInstance::getDevice: Device memory should have exactly one "
         "device.");

  return m_addressable_by_devices[0];
}

namespace internal {

PJRT_Error *onMemoryId(PJRT_Memory_Id_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_Id");

  args->id = MemoryInstance::unwrap(args->memory)->getId();

  return nullptr;
}

PJRT_Error *onMemoryKind(PJRT_Memory_Kind_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_Kind");

  MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  args->kind = memory_instance->getMemoryKind().data();
  args->kind_size = memory_instance->getMemoryKind().size();

  return nullptr;
}

PJRT_Error *onMemoryKindId(PJRT_Memory_Kind_Id_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_Kind_Id");

  MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  args->kind_id = memory_instance->isHostMemory() ? 0 : 1;

  return nullptr;
}

PJRT_Error *onMemoryDebugString(PJRT_Memory_DebugString_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_DebugString");

  MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  args->debug_string = memory_instance->getDebugString().data();
  args->debug_string_size = memory_instance->getDebugString().size();

  return nullptr;
}

PJRT_Error *onMemoryToString(PJRT_Memory_ToString_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_ToString");

  MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  args->to_string = memory_instance->getDebugString().data();
  args->to_string_size = memory_instance->getDebugString().size();

  return nullptr;
}

PJRT_Error *
onMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_AddressableByDevices");

  const MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  const std::vector<DeviceInstance *> &addressable_by_devices =
      memory_instance->getAddressableByDevices();

  args->devices =
      reinterpret_cast<PJRT_Device *const *>(addressable_by_devices.data());
  args->num_devices = addressable_by_devices.size();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

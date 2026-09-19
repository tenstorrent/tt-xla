// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/memory_instance.h"

// c++ standard library includes
#include <cstring>

// tracy includes
#include "tracy/Tracy.hpp"

// tt-xla includes
#include "utils/assert.h"
#include "utils/logging.h"

namespace tt::pjrt {

const std::string MemoryInstance::c_host_memory_kind_name = "pinned_host";
const std::string MemoryInstance::c_device_memory_kind_name = "device";

namespace {

// Backs the PJRT_Memory vtable `get_user_data` slot: forwards to the owning
// MemoryInstance.
void *memoryGetUserData(PJRT_Memory *memory, const void *key) {
  return MemoryInstance::unwrap(memory)->getUserData(key);
}

// Backs the PJRT_Memory vtable `set_user_data` slot: forwards to the owning
// MemoryInstance.
void memorySetUserData(PJRT_Memory *memory, const void *key, void *data,
                       void (*dtor)(void *)) {
  MemoryInstance::unwrap(memory)->setUserData(key, data, dtor);
}

// The single PJRT_Memory vtable shared by every MemoryInstance. It only exposes
// the user-data accessors the framework needs to associate its own per-memory
// state with our PJRT_Memory objects.
const PJRT_Memory_FunctionTable kMemoryFunctionTable = {
    /*struct_size=*/PJRT_Memory_FunctionTable_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*instance_struct_size=*/PJRT_Memory_STRUCT_SIZE,
    /*get_user_data=*/memoryGetUserData,
    /*set_user_data=*/memorySetUserData,
};

} // namespace

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
    : m_vtable(&kMemoryFunctionTable),
      m_addressable_by_devices(addressable_by_devices),
      m_is_host_memory(is_host_memory), m_id(id) {
  m_debug_string =
      "MemoryInstance: " + std::to_string(id) + " (" + getMemoryKind() + ")";
}

MemoryInstance::~MemoryInstance() {
  std::lock_guard<std::mutex> lock(m_user_data_mutex);
  for (auto &[key, value] : m_user_data) {
    auto &[data, dtor] = value;
    if (dtor != nullptr) {
      dtor(data);
    }
  }
}

void *MemoryInstance::getUserData(const void *key) {
  std::lock_guard<std::mutex> lock(m_user_data_mutex);
  auto it = m_user_data.find(key);
  return it == m_user_data.end() ? nullptr : it->second.first;
}

void MemoryInstance::setUserData(const void *key, void *data,
                                 void (*dtor)(void *)) {
  std::lock_guard<std::mutex> lock(m_user_data_mutex);
  auto it = m_user_data.find(key);
  if (it != m_user_data.end() && it->second.second != nullptr &&
      it->second.first != nullptr) {
    // Destroy the previous value before overwriting or erasing it.
    it->second.second(it->second.first);
  }

  if (data == nullptr) {
    // A null value clears the entry, matching the framework's convention.
    if (it != m_user_data.end()) {
      m_user_data.erase(it);
    }
    return;
  }

  if (it != m_user_data.end()) {
    it->second = {data, dtor};
  } else {
    m_user_data.emplace(key, std::make_pair(data, dtor));
  }
}

DeviceInstance *MemoryInstance::getDevice() {
  DLOG_F(LOG_DEBUG, "MemoryInstance::getDevice");

  if (isHostMemory()) {
    DLOG_F(WARNING,
           "MemoryInstance::getDevice: Host memory does not have a device.");

    return nullptr;
  }

  TT_FATAL(m_addressable_by_devices.size() == 1,
           "MemoryInstance::getDevice: Device memory should have exactly one "
           "device: m_addressable_by_devices.size()={}",
           m_addressable_by_devices.size());

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

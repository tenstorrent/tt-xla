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

#include "common/status.h"
#include <cstring>

namespace tt::pjrt {

std::unique_ptr<MemoryInstance> MemoryInstance::createInstance(
    std::vector<DeviceInstance *> &addressable_by_devices,
    std::string memory_kind) {
  struct make_unique_enabler : public MemoryInstance {
    make_unique_enabler(std::vector<DeviceInstance *> &addressable_by_devices,
                        std::string memory_kind)
        : MemoryInstance(addressable_by_devices, memory_kind) {}
  };
  return std::make_unique<make_unique_enabler>(addressable_by_devices,
                                               memory_kind);
}

void MemoryInstance::bindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");

  api->PJRT_Memory_AddressableByDevices =
      internal::onMemoryAddressableByDevices;
  api->PJRT_Memory_Kind = internal::onMemoryKind;
}

namespace internal {

PJRT_Error *
onMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args) {
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_AddressableByDevices");
  args->num_devices =
      MemoryInstance::Unwrap(args->memory)->getAddressableByDevices().size();
  const std::vector<DeviceInstance *> &addressable_by_devices =
      MemoryInstance::Unwrap(args->memory)->getAddressableByDevices();
  args->devices = const_cast<PJRT_Device **>(
      reinterpret_cast<PJRT_Device *const *>(addressable_by_devices.data()));
  return nullptr;
}

PJRT_Error *onMemoryKind(PJRT_Memory_Kind_Args *args) {
  DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_Kind");
  MemoryInstance *memory_instance = MemoryInstance::Unwrap(args->memory);
  args->kind = memory_instance->getMemoryKind().data();
  args->kind_size = memory_instance->getMemoryKind().size();
  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

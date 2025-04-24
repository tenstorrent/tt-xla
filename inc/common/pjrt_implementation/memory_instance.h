// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <string>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance;

class MemoryInstance {

public:
  static std::unique_ptr<MemoryInstance>
  createInstance(std::vector<DeviceInstance *> &addressable_by_devices,
                 size_t id, std::string memory_kind);

  operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }

  // Binds PJRT API functions implementation related to PJRT_LoadedExecutable
  // structure.
  static void bindApi(PJRT_Api *api);

  static MemoryInstance *Unwrap(PJRT_Memory *device_description) {
    return reinterpret_cast<MemoryInstance *>(device_description);
  }

  const std::vector<DeviceInstance *> &getAddressableByDevices() const {
    return m_addressable_by_devices;
  }

  const std::string &getMemoryKind() const { return m_memory_kind; }

  const size_t getId() const { return m_id; }

  const std::string &getDebugString() const { return m_debug_string; }

private:
  MemoryInstance(std::vector<DeviceInstance *> &addressable_by_devices,
                 size_t id, std::string memory_kind);

  // List of devices that can access this memory.
  std::vector<DeviceInstance *> m_addressable_by_devices;

  // String representing the kind of memory, can be 'tt_host' or 'tt_device'.
  std::string m_memory_kind;

  // Id of the memory.
  size_t m_id;

  // Debug string of the memory.
  std::string m_debug_string;
};

namespace internal {

// Implements PJRT_Memory_AddressableByDevices API function.
PJRT_Error *
onMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args);

// Implements PJRT_Memory_Kind API function.
PJRT_Error *onMemoryKind(PJRT_Memory_Kind_Args *args);

// Implements PJRT_Memory_Id API function.
PJRT_Error *onMemoryId(PJRT_Memory_Id_Args *args);

// Implements PJRT_Memory_DebugString API function.
PJRT_Error *onMemoryDebugString(PJRT_Memory_DebugString_Args *args);

// Implements PJRT_Memory_ToString API function.
PJRT_Error *onMemoryToString(PJRT_Memory_ToString_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif

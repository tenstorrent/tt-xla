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

// Represents PJRT_Memory structure and the functionality around it.
class MemoryInstance {

public:
  // Creates a new memory instance.
  static std::unique_ptr<MemoryInstance>
  createInstance(std::vector<DeviceInstance *> &addressable_by_devices,
                 size_t id, std::string memory_kind);

  // Binds PJRT API functions implementation related to PJRT_Memory structure.
  static void bindApi(PJRT_Api *api);

  // Casts the PJRT_Memory pointer to MemoryInstance pointer.
  static MemoryInstance *unwrap(PJRT_Memory *device_description) {
    return reinterpret_cast<MemoryInstance *>(device_description);
  }

  // Casts this memory instance to PJRT_Memory pointer.
  operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }

  // Gets the list of devices that can address this memory.
  const std::vector<DeviceInstance *> &getAddressableByDevices() const {
    return m_addressable_by_devices;
  }

  // Gets the string representing the kind of memory.
  // It can be 'tt_host' or 'tt_device'.
  const std::string &getMemoryKind() const { return m_memory_kind; }

  // Gets the id of the memory (host - 0, device - (1, 2, ...)).
  const size_t getId() const { return m_id; }

  // Gets the debug string representing the memory.
  const std::string &getDebugString() const { return m_debug_string; }

  // Gets the device that the memory is on.
  DeviceInstance *getDevice() const;

  // String that represents the host memory kind.
  static const std::string host_memory_kind_name;

  // String that represents the device memory kind.
  static const std::string device_memory_kind_name;

private:
  // Private constructor to prevent direct instantiation.
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

// Implements PJRT_Memory_Kind_Id API function.
PJRT_Error *onMemoryKindId(PJRT_Memory_Kind_Id_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif

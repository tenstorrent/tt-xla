// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MEMORY_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MEMORY_INSTANCE_H_

// c++ standard library includes
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

class DeviceInstance;

// Represents PJRT_Memory structure and the functionality around it.
class MemoryInstance {

public:
  // Creates a new memory instance.
  static std::unique_ptr<MemoryInstance>
  createInstance(std::vector<DeviceInstance *> &addressable_by_devices,
                 size_t id, bool is_host_memory);

  // Binds PJRT API functions implementation related to PJRT_Memory structure.
  static void bindApi(PJRT_Api *api);

  // Destroys the memory instance, invoking the destructors of any user data
  // that the framework attached via the PJRT_Memory vtable.
  ~MemoryInstance();

  // Casts this memory instance to PJRT_Memory pointer.
  operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }

  // Casts the PJRT_Memory pointer to MemoryInstance pointer.
  static MemoryInstance *unwrap(PJRT_Memory *memory) {
    return reinterpret_cast<MemoryInstance *>(memory);
  }

  // Returns the user data attached under `key`, or nullptr if none. Backs the
  // PJRT_Memory vtable `get_user_data` slot.
  void *getUserData(const void *key);

  // Attaches `data` (owned via `dtor`) under `key`, replacing and destroying any
  // previous value stored under the same key. Backs the PJRT_Memory vtable
  // `set_user_data` slot.
  void setUserData(const void *key, void *data, void (*dtor)(void *));

  // Gets the list of devices that can address this memory.
  const std::vector<DeviceInstance *> &getAddressableByDevices() const {
    return m_addressable_by_devices;
  }

  // Checks if the memory is host memory.
  bool isHostMemory() const { return m_is_host_memory; }

  // Gets the string representing the kind of memory.
  // It can be 'tt_host' or 'tt_device'.
  const std::string &getMemoryKind() const {
    return m_is_host_memory ? MemoryInstance::c_host_memory_kind_name
                            : MemoryInstance::c_device_memory_kind_name;
  }

  // Gets the id of the memory (host - 0, device - (1, 2, ...)).
  int getId() const { return m_id; }

  // Gets the debug string representing the memory.
  const std::string &getDebugString() const { return m_debug_string; }

  // Gets the device that the memory is on.
  DeviceInstance *getDevice();

  // String that represents the host memory kind.
  static const std::string c_host_memory_kind_name;

  // String that represents the device memory kind.
  static const std::string c_device_memory_kind_name;

private:
  // Private constructor to prevent direct instantiation.
  MemoryInstance(std::vector<DeviceInstance *> &addressable_by_devices,
                 size_t id, bool is_host_memory);

  // Pointer to the PJRT_Memory vtable. MUST be the first member: the PJRT C API
  // models PJRT_Memory as a struct whose sole field is a leading pointer to a
  // PJRT_Memory_FunctionTable.
  const PJRT_Memory_FunctionTable *m_vtable;

  // List of devices that can access this memory.
  std::vector<DeviceInstance *> m_addressable_by_devices;

  // Denotes if the memory is host memory.
  bool m_is_host_memory;

  // Id of the memory.
  int m_id;

  // Debug string of the memory.
  std::string m_debug_string;

  // User data attached by the framework through the PJRT_Memory vtable, keyed by
  // an opaque pointer. Each entry owns its value via the stored destructor.
  std::unordered_map<const void *, std::pair<void *, void (*)(void *)>>
      m_user_data;

  // Guards `m_user_data`; the framework may access user data from multiple
  // threads.
  std::mutex m_user_data_mutex;
};

namespace internal {

// Implements PJRT_Memory_Id API function.
PJRT_Error *onMemoryId(PJRT_Memory_Id_Args *args);

// Implements PJRT_Memory_Kind API function.
PJRT_Error *onMemoryKind(PJRT_Memory_Kind_Args *args);

// Implements PJRT_Memory_Kind_Id API function.
PJRT_Error *onMemoryKindId(PJRT_Memory_Kind_Id_Args *args);

// Implements PJRT_Memory_DebugString API function.
PJRT_Error *onMemoryDebugString(PJRT_Memory_DebugString_Args *args);

// Implements PJRT_Memory_ToString API function.
PJRT_Error *onMemoryToString(PJRT_Memory_ToString_Args *args);

// Implements PJRT_Memory_AddressableByDevices API function.
PJRT_Error *
onMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MEMORY_INSTANCE_H_

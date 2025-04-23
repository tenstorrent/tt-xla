// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "xla/pjrt/c/pjrt_c_api.h"

#include <string>
#include <vector>

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MEMORY_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance;

class MemoryInstance {

public:
    MemoryInstance(std::vector<DeviceInstance *>& addressable_by_devices, std::string memory_kind) : 
        m_addressable_by_devices(addressable_by_devices), m_memory_kind(memory_kind) {}
    ~MemoryInstance() = default;

    operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }

    // Binds PJRT API functions implementation related to PJRT_LoadedExecutable
    // structure.
    static void bindApi(PJRT_Api *api);

    static MemoryInstance *Unwrap(PJRT_Memory *device_description) {
        return reinterpret_cast<MemoryInstance *>(device_description);
    }

    const std::vector<DeviceInstance *> &getAddressableByDevices() const { return m_addressable_by_devices; }

    const std::string &getMemoryKind() const { return m_memory_kind; }

private:
    // List of devices that can access this memory.
    std::vector<DeviceInstance *> m_addressable_by_devices;

    std::string m_memory_kind;
};

namespace internal {

// Implements PJRT_Memory_AddressableByDevices API function.
PJRT_Error *onMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args);

// Implements PJRT_Memory_Kind API function.
PJRT_Error *onMemoryKind(PJRT_Memory_Kind_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif
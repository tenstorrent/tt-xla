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
    MemoryInstance(std::vector<DeviceInstance *>& addressable_by_devices, int memory_id, std::string memory_kind) : 
        m_addressable_by_devices(addressable_by_devices), m_memory_id(memory_id), m_memory_kind(memory_kind) {}
    ~MemoryInstance();

    operator PJRT_Memory *() { return reinterpret_cast<PJRT_Memory *>(this); }

    static void BindApi(PJRT_Api *api);

    static MemoryInstance *Unwrap(PJRT_Memory *device_description) {
        return reinterpret_cast<MemoryInstance *>(device_description);
    }

    const std::vector<DeviceInstance *> &addressable_by_devices() const { return m_addressable_by_devices; }

    const std::string &memory_kind() const { return m_memory_kind; }

    const int getMemoryId() const { return m_memory_id; }

private:
    // List of devices that can access this memory.
    std::vector<DeviceInstance *>& m_addressable_by_devices;

    // Id of the memory instance.
    int m_memory_id;

    std::string m_memory_kind;
};

} // namespace tt::pjrt

#endif
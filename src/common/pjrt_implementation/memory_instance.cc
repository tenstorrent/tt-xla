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
#include <iostream>

namespace tt::pjrt {

void MemoryInstance::BindApi(PJRT_Api *api) {
    DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");

    api->PJRT_Memory_AddressableByDevices = 
        +[](PJRT_Memory_AddressableByDevices_Args *args) -> PJRT_Error * {
            DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_AddressableByDevices");
            args->num_devices = MemoryInstance::Unwrap(args->memory)->addressable_by_devices().size();
            std::cerr << "memory=" << args->memory << std::endl;
            std::cerr << "num_devices=" << args->num_devices << std::endl;
            args->devices = const_cast<PJRT_Device **>(reinterpret_cast<PJRT_Device *const *>(MemoryInstance::Unwrap(args->memory)->addressable_by_devices().data()));
            return nullptr;
    };

    api->PJRT_Memory_Kind =
        +[](PJRT_Memory_Kind_Args *args) -> PJRT_Error * {
            DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_Kind");
            MemoryInstance *memory_instance = MemoryInstance::Unwrap(args->memory);
            args->kind = memory_instance->memory_kind().data();
            args->kind_size = memory_instance->memory_kind().size();
            return nullptr;
    };
}

} // namespace tt::pjrt

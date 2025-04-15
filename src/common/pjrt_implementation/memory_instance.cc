// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

void MemoryInstance::BindApi(PJRT_Api *api) {
    DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");

    api->PJRT_Memory_AddressableByDevices = 
        +[](PJRT_Memory_AddressableByDevices_Args *args) -> PJRT_Error * {
            args->num_devices = MemoryInstance::Unwrap(args->memory)->addressable_by_devices().size();
            DLOG_F(LOG_DEBUG, "MemoryInstance::PJRT_Memory_AddressableByDevices num devices: %ld", args->num_devices);
            args->devices = const_cast<PJRT_Device **>(reinterpret_cast<PJRT_Device *const *>(MemoryInstance::Unwrap(args->memory)->addressable_by_devices().data()));
            return nullptr;
    };
}

}  // namespace tt::pjrt
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api_bindings.h"

#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_description.h"
#include "api/device_instance.h"
#include "api/error_instance.h"
#include "api/event_instance.h"
#include "api/executable_instance.h"
#include "api/loaded_executable_instance.h"
#include "api/memory_instance.h"
#include "api/plugin_attributes.h"
#include "utils/logging.h"

namespace tt::pjrt {

void bindApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  initializeLogging();

  bindUndefineds(api);

  // Bind by object types.
  BufferInstance::bindApi(api);
  ClientInstance::bindApi(api);
  DeviceDescription::bindApi(api);
  DeviceInstance::bindApi(api);
  EventInstance::bindApi(api);
  ErrorInstance::bindApi(api);
  ExecutableInstance::bindApi(api);
  LoadedExecutableInstance::bindApi(api);
  MemoryInstance::bindApi(api);
  PluginAttributes::bindApi(api);
}

void bindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(WARNING, "STUB: " #API);                                            \
    return (decltype(api->API(args)))*ErrorInstance::makeError(                \
               tt_pjrt_status::kUnimplemented)                                 \
        .release();                                                            \
  }

#include "stubs.inc"
}

} // namespace tt::pjrt

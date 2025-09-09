// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "entry/api_bindings.h"

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

void BindMonomorphicApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  // This is a bare implementation throwing UNDEFINED errors. This way new
  // functions will not segmentation fault on invocation.
  BindUndefineds(api);

  api->PJRT_Plugin_Initialize =
      +[](PJRT_Plugin_Initialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Initialize");
    return nullptr;
  };

  api->PJRT_Plugin_Attributes =
      +[](PJRT_Plugin_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");
    args->num_attributes = 0;
    return nullptr;
  };

  api->PJRT_Plugin_Attributes = InitializePluginAttributes;

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
}

void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(WARNING, "STUB: " #API);                                            \
    return (decltype(api->API(args)))*ErrorInstance::makeError(                \
               tt_pjrt_status::kUnimplemented)                                 \
        .release();                                                            \
  }

#include "stubs/stubs.inc"
}

PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");

  static std::unique_ptr<PJRTPluginAttributes> s_plugin_attributes =
      std::make_unique<PJRTPluginAttributes>();
  args->attributes = s_plugin_attributes->getAttributes();
  args->num_attributes = s_plugin_attributes->getNumAttributes();

  return nullptr;
}

void BindApi(PJRT_Api *api) {
  BindMonomorphicApi(api);

  // Bind polymorphic entry-points.
  api->PJRT_Client_Create = +[](PJRT_Client_Create_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Client_Create");

    // Populate config_vars() from the client create_options.
    for (size_t i = 0; i < args->num_options; ++i) {
      DLOG_F(WARNING, "Unused config var: %s", args->create_options[i].name);
    }

    InitializeLogging();

    auto client = std::make_unique<ClientInstance>();
    auto *error = client->Initialize();
    if (error)
      return error;

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client *>(client.release());
    return nullptr;
  };
}

} // namespace tt::pjrt

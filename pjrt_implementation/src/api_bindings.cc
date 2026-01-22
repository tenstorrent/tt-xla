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

#include <string>
#include <unordered_map>

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
#include "api/tt_pjrt_device_options_extension.h"
#include "utils/logging.h"

namespace tt::pjrt {

namespace {

// Global extension instance (static lifetime).
PJRT_TT_SetDeviceOptions_Extension g_set_device_options_extension;

// Handler for PJRT_TT_SetDeviceOptions.
PJRT_Error *onSetDeviceOptions(PJRT_TT_SetDeviceOptions_Args *args) {
  DLOG_F(LOG_DEBUG, "PJRT_TT_SetDeviceOptions called for device %d with %zu "
                    "options",
         args->device_id, args->num_device_options);

  // Use the global singleton to get the client instance.
  ClientInstance *client = GlobalClientInstanceSingleton::getClientInstance();
  if (client == nullptr) {
    DLOG_F(ERROR, "PJRT_TT_SetDeviceOptions: client not initialized");
    return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument).release();
  }

  // Convert PJRT_NamedValue array to unordered_map.
  std::unordered_map<std::string, std::string> options;
  for (size_t i = 0; i < args->num_device_options; ++i) {
    const auto &opt = args->device_options[i];
    std::string key(opt.name, opt.name_size);
    if (opt.type == PJRT_NamedValue_kString) {
      options[key] = std::string(opt.string_value, opt.value_size);
    } else if (opt.type == PJRT_NamedValue_kBool) {
      options[key] = opt.bool_value ? "true" : "false";
    } else if (opt.type == PJRT_NamedValue_kInt64) {
      options[key] = std::to_string(opt.int64_value);
    }
  }

  // Set options on the client for the specified device.
  client->setCustomDeviceOptions(args->device_id, options);

  return nullptr;
}

} // namespace

void bindApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  initializeLogging();

  // Initialize the SetDeviceOptions extension.
  g_set_device_options_extension.struct_size =
      PJRT_TT_SetDeviceOptions_Extension_STRUCT_SIZE;
  g_set_device_options_extension.type =
      PJRT_Extension_Type_TT_SetDeviceOptions;
  g_set_device_options_extension.next = nullptr;
  g_set_device_options_extension.set_device_options = onSetDeviceOptions;

  // Link extension to the API.
  api->extension_start =
      reinterpret_cast<PJRT_Extension_Base *>(&g_set_device_options_extension);

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

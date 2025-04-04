// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/api_bindings.h"

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_description.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"
#include "common/plugin_attributes.h"

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
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::bindApi(api);
  EventInstance::bindApi(api);
  ErrorInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(WARNING, "STUB: " #API);                                            \
    return (decltype(api->API(args)))ErrorInstance::MakeError(                 \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "common/pjrt_implementation/stubs.inc"
}

PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");

  static std::unique_ptr<PJRTPluginAttributes> s_plugin_attributes =
      std::make_unique<PJRTPluginAttributes>();
  args->attributes = s_plugin_attributes->getAttributes();
  args->num_attributes = s_plugin_attributes->getNumAttributes();

  return nullptr;
}

} // namespace tt::pjrt

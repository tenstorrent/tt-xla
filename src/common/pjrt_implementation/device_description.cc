// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/device_description.h"

#include "common/status.h"

namespace tt::pjrt {

DeviceDescription::~DeviceDescription() = default;

void DeviceDescription::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::BindApi");
  api->PJRT_DeviceDescription_Id =
      +[](PJRT_DeviceDescription_Id_Args *args) -> PJRT_Error * {
    args->id = DeviceDescription::Unwrap(args->device_description)->client_id();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ProcessIndex =
      +[](PJRT_DeviceDescription_ProcessIndex_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ProcessIndex");
    args->process_index =
        DeviceDescription::Unwrap(args->device_description)->process_index();
    return nullptr;
  };
  api->PJRT_DeviceDescription_Attributes =
      +[](PJRT_DeviceDescription_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Attributes");
    // TODO: Implement something.
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_DeviceDescription_Kind =
      +[](PJRT_DeviceDescription_Kind_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Kind");
    std::string_view sv =
        DeviceDescription::Unwrap(args->device_description)->kind_string();
    args->device_kind = sv.data();
    args->device_kind_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_DebugString =
      +[](PJRT_DeviceDescription_DebugString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_DebugString");
    std::string_view sv =
        DeviceDescription::Unwrap(args->device_description)->debug_string();
    args->debug_string = sv.data();
    args->debug_string_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ToString =
      +[](PJRT_DeviceDescription_ToString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ToString");
    std::string_view sv = DeviceDescription::Unwrap(args->device_description)->to_string();
    args->to_string = sv.data();
    args->to_string_size = sv.size();
    return nullptr;
  };
}

} // namespace tt::pjrt

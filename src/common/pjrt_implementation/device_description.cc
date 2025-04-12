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

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt {

DeviceDescription::DeviceDescription(int32_t device_id, tt::target::Arch arch)
    : m_device_id(device_id), m_device_kind(tt::target::EnumNameArch(arch)) {
  std::stringstream ss;
  ss << "TTDevice(id=" << getDeviceId() << ", arch=" << m_device_kind << ")";
  m_user_string = ss.str();
}

DeviceDescription::~DeviceDescription() = default;

void DeviceDescription::bindApi(PJRT_Api *api) {
  api->PJRT_DeviceDescription_Id =
      +[](PJRT_DeviceDescription_Id_Args *args) -> PJRT_Error * {
    args->id =
        DeviceDescription::unwrap(args->device_description)->getDeviceId();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ProcessIndex =
      +[](PJRT_DeviceDescription_ProcessIndex_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ProcessIndex");
    args->process_index =
        DeviceDescription::unwrap(args->device_description)->getProcessIndex();
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
    const std::string &device_kind =
        DeviceDescription::unwrap(args->device_description)->getDeviceKind();
    args->device_kind = device_kind.data();
    args->device_kind_size = device_kind.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_DebugString =
      +[](PJRT_DeviceDescription_DebugString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_DebugString");
    const std::string &debug_str =
        DeviceDescription::unwrap(args->device_description)->toDebugString();
    args->debug_string = debug_str.data();
    args->debug_string_size = debug_str.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ToString =
      +[](PJRT_DeviceDescription_ToString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ToString");
    const std::string &description_str =
        DeviceDescription::unwrap(args->device_description)->toString();
    args->to_string = description_str.data();
    args->to_string_size = description_str.size();
    return nullptr;
  };
}

} // namespace tt::pjrt

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

// c++ standard library includes
#include <sstream>

// tracy includes
#include <tracy/Tracy.hpp>

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt {

DeviceDescription::DeviceDescription(int32_t device_id, tt::target::Arch arch)
    : m_device_id(device_id), m_process_index(0),
      m_device_kind(tt::target::EnumNameArch(arch)) {
  std::stringstream ss;
  ss << "TTDevice(id=" << m_device_id << ", arch=" << m_device_kind << ")";
  m_user_string = ss.str();
}

void DeviceDescription::bindApi(PJRT_Api *api) {
  api->PJRT_DeviceDescription_Id = internal::onDeviceDescriptionId;
  api->PJRT_DeviceDescription_ProcessIndex =
      internal::onDeviceDescriptionProcessIndex;
  api->PJRT_DeviceDescription_Attributes =
      internal::onDeviceDescriptionAttributes;
  api->PJRT_DeviceDescription_Kind = internal::onDeviceDescriptionKind;
  api->PJRT_DeviceDescription_DebugString =
      internal::onDeviceDescriptionDebugString;
  api->PJRT_DeviceDescription_ToString = internal::onDeviceDescriptionToString;
}

namespace internal {

PJRT_Error *onDeviceDescriptionId(PJRT_DeviceDescription_Id_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Id");

  args->id = DeviceDescription::unwrap(args->device_description)->getDeviceId();

  return nullptr;
}

PJRT_Error *onDeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ProcessIndex");

  args->process_index =
      DeviceDescription::unwrap(args->device_description)->getProcessIndex();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionAttributes(PJRT_DeviceDescription_Attributes_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Attributes");

  // We don't set any device attributes currently.
  args->num_attributes = 0;
  args->attributes = nullptr;

  return nullptr;
}

PJRT_Error *onDeviceDescriptionKind(PJRT_DeviceDescription_Kind_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Kind");

  const std::string &device_kind =
      DeviceDescription::unwrap(args->device_description)->getDeviceKind();

  args->device_kind = device_kind.data();
  args->device_kind_size = device_kind.size();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionDebugString(PJRT_DeviceDescription_DebugString_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_DebugString");

  const std::string &debug_str =
      DeviceDescription::unwrap(args->device_description)->toDebugString();

  args->debug_string = debug_str.data();
  args->debug_string_size = debug_str.size();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionToString(PJRT_DeviceDescription_ToString_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ToString");

  const std::string &description_str =
      DeviceDescription::unwrap(args->device_description)->toString();

  args->to_string = description_str.data();
  args->to_string_size = description_str.size();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

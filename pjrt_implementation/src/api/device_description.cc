// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/device_description.h"

// c++ standard library includes
#include <sstream>

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt {

DeviceDescription::DeviceDescription(int32_t device_id, tt::target::Arch arch)
    : m_device_id(device_id), m_process_index(0),
      m_device_kind(tt::target::EnumNameArch(arch)) {
  std::stringstream ss;
  ss << "TTDevice(id=" << m_device_id << ", arch=" << m_device_kind << ")";
  m_user_string = ss.str();
}

void DeviceDescription::setCustomDeviceOptions(
    const std::unordered_map<std::string, std::string> &options) {
  m_custom_device_options = options;

  // Rebuild the PJRT attributes array from the options.
  m_attributes.clear();
  m_attributes.reserve(options.size());

  for (const auto &[key, value] : m_custom_device_options) {
    PJRT_NamedValue attr;
    attr.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    attr.extension_start = nullptr;
    attr.name = key.c_str();
    attr.name_size = key.size();
    attr.type = PJRT_NamedValue_kString;
    attr.string_value = value.c_str();
    attr.value_size = value.size();
    m_attributes.push_back(attr);
  }

  DLOG_F(LOG_DEBUG,
         "DeviceDescription::setCustomDeviceOptions - set %zu options for "
         "device %d",
         options.size(), m_device_id);
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
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Id");

  args->id = DeviceDescription::unwrap(args->device_description)->getDeviceId();

  return nullptr;
}

PJRT_Error *onDeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ProcessIndex");

  args->process_index =
      DeviceDescription::unwrap(args->device_description)->getProcessIndex();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionAttributes(PJRT_DeviceDescription_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Attributes");

  DeviceDescription *desc =
      DeviceDescription::unwrap(args->device_description);
  args->num_attributes = desc->getNumAttributes();
  args->attributes =
      desc->getNumAttributes() > 0 ? desc->getAttributes().data() : nullptr;

  return nullptr;
}

PJRT_Error *onDeviceDescriptionKind(PJRT_DeviceDescription_Kind_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Kind");

  const std::string &device_kind =
      DeviceDescription::unwrap(args->device_description)->getDeviceKind();

  args->device_kind = device_kind.data();
  args->device_kind_size = device_kind.size();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionDebugString(PJRT_DeviceDescription_DebugString_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_DebugString");

  const std::string &debug_str =
      DeviceDescription::unwrap(args->device_description)->toDebugString();

  args->debug_string = debug_str.data();
  args->debug_string_size = debug_str.size();

  return nullptr;
}

PJRT_Error *
onDeviceDescriptionToString(PJRT_DeviceDescription_ToString_Args *args) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ToString");

  const std::string &description_str =
      DeviceDescription::unwrap(args->device_description)->toString();

  args->to_string = description_str.data();
  args->to_string_size = description_str.size();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

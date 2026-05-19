// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/topology_description.h"

#include <utility>

#include "api/device_description.h"
#include "utils/logging.h"

namespace tt::pjrt {

TopologyDescription::TopologyDescription(
    std::string platform_name, std::string platform_version,
    std::vector<DeviceDescription *> device_descriptions)
    : m_platform_name(std::move(platform_name)),
      m_platform_version(std::move(platform_version)) {
  m_device_descriptions_raw.reserve(device_descriptions.size());
  for (DeviceDescription *description : device_descriptions) {
    m_device_descriptions_raw.push_back(
        reinterpret_cast<PJRT_DeviceDescription *>(description));
  }
}

void TopologyDescription::bindApi(PJRT_Api *api) {
  api->PJRT_TopologyDescription_Destroy =
      internal::onTopologyDescriptionDestroy;
  api->PJRT_TopologyDescription_PlatformName =
      internal::onTopologyDescriptionPlatformName;
  api->PJRT_TopologyDescription_PlatformVersion =
      internal::onTopologyDescriptionPlatformVersion;
  api->PJRT_TopologyDescription_GetDeviceDescriptions =
      internal::onTopologyDescriptionGetDeviceDescriptions;
  api->PJRT_TopologyDescription_Attributes =
      internal::onTopologyDescriptionAttributes;
  api->PJRT_TopologyDescription_Fingerprint =
      internal::onTopologyDescriptionFingerprint;
}

namespace internal {

// The topology returned by PJRT_Client_TopologyDescription is owned by the
// client (see pjrt_c_api.h:601). Destroy is therefore a no-op here — the
// client cleans it up on PJRT_Client_Destroy. A future
// PJRT_TopologyDescription_Create path (AOT compilation) would need a
// self-owned variant that actually frees.
PJRT_Error *
onTopologyDescriptionDestroy(PJRT_TopologyDescription_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "TopologyDescription::PJRT_TopologyDescription_Destroy");
  return nullptr;
}

PJRT_Error *onTopologyDescriptionPlatformName(
    PJRT_TopologyDescription_PlatformName_Args *args) {
  DLOG_F(LOG_DEBUG,
         "TopologyDescription::PJRT_TopologyDescription_PlatformName");

  const std::string &name =
      TopologyDescription::unwrap(args->topology)->getPlatformName();
  args->platform_name = name.data();
  args->platform_name_size = name.size();
  return nullptr;
}

PJRT_Error *onTopologyDescriptionPlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args *args) {
  DLOG_F(LOG_DEBUG,
         "TopologyDescription::PJRT_TopologyDescription_PlatformVersion");

  const std::string &version =
      TopologyDescription::unwrap(args->topology)->getPlatformVersion();
  args->platform_version = version.data();
  args->platform_version_size = version.size();
  return nullptr;
}

PJRT_Error *onTopologyDescriptionGetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args *args) {
  DLOG_F(LOG_DEBUG,
         "TopologyDescription::PJRT_TopologyDescription_GetDeviceDescriptions");

  const std::vector<PJRT_DeviceDescription *> &descriptions =
      TopologyDescription::unwrap(args->topology)->getDeviceDescriptionsRaw();
  args->descriptions = descriptions.data();
  args->num_descriptions = descriptions.size();
  return nullptr;
}

PJRT_Error *onTopologyDescriptionAttributes(
    PJRT_TopologyDescription_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "TopologyDescription::PJRT_TopologyDescription_Attributes");

  const std::vector<PJRT_NamedValue> &attributes =
      TopologyDescription::unwrap(args->topology)->getAttributes();
  args->attributes = attributes.data();
  args->num_attributes = attributes.size();
  return nullptr;
}

PJRT_Error *onTopologyDescriptionFingerprint(
    PJRT_TopologyDescription_Fingerprint_Args *args) {
  DLOG_F(LOG_DEBUG,
         "TopologyDescription::PJRT_TopologyDescription_Fingerprint");

  // Fingerprint is used as an AOT compilation cache key; tt-xla does not
  // support AOT compilation yet, so a constant value is sufficient — every
  // client sees the same topology.
  args->fingerprint = 0;
  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

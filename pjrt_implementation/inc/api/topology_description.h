// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TOPOLOGY_DESCRIPTION_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TOPOLOGY_DESCRIPTION_H_

// c++ standard library includes
#include <string>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

class ClientInstance;
class DeviceDescription;

// Represents PJRT_TopologyDescription structure and the functionality around
// it. A TopologyDescription is owned by the client that created it (see
// PJRT_Client_TopologyDescription), and its lifetime matches the client's.
class TopologyDescription {
public:
  TopologyDescription(std::string platform_name, std::string platform_version,
                      std::vector<DeviceDescription *> device_descriptions);

  // Binds PJRT API functions implementation related to PJRT_TopologyDescription
  // structure.
  static void bindApi(PJRT_Api *api);

  operator PJRT_TopologyDescription *() {
    return reinterpret_cast<PJRT_TopologyDescription *>(this);
  }

  static TopologyDescription *unwrap(PJRT_TopologyDescription *topology) {
    return reinterpret_cast<TopologyDescription *>(topology);
  }

  static const TopologyDescription *
  unwrap(const PJRT_TopologyDescription *topology) {
    return reinterpret_cast<const TopologyDescription *>(topology);
  }

  const std::string &getPlatformName() const { return m_platform_name; }
  const std::string &getPlatformVersion() const { return m_platform_version; }

  // Raw PJRT_DeviceDescription pointers backed by the owning client's
  // DeviceInstance objects. Lifetime matches the client.
  const std::vector<PJRT_DeviceDescription *> &
  getDeviceDescriptionsRaw() const {
    return m_device_descriptions_raw;
  }

  const std::vector<PJRT_NamedValue> &getAttributes() const {
    return m_attributes;
  }

private:
  std::string m_platform_name;
  std::string m_platform_version;
  // Cached raw pointers ready to return from
  // PJRT_TopologyDescription_GetDeviceDescriptions. Owned by the client.
  std::vector<PJRT_DeviceDescription *> m_device_descriptions_raw;
  // Empty attribute list for now; reserved for platform-specific topology
  // info if/when downstream needs it.
  std::vector<PJRT_NamedValue> m_attributes;
};

namespace internal {

PJRT_Error *
onTopologyDescriptionDestroy(PJRT_TopologyDescription_Destroy_Args *args);

PJRT_Error *onTopologyDescriptionPlatformName(
    PJRT_TopologyDescription_PlatformName_Args *args);

PJRT_Error *onTopologyDescriptionPlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args *args);

PJRT_Error *onTopologyDescriptionGetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args *args);

PJRT_Error *
onTopologyDescriptionAttributes(PJRT_TopologyDescription_Attributes_Args *args);

PJRT_Error *onTopologyDescriptionFingerprint(
    PJRT_TopologyDescription_Fingerprint_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TOPOLOGY_DESCRIPTION_H_

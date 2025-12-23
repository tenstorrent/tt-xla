// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/plugin_attributes.h"

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt {

StableHLOVersionAttribute::StableHLOVersionAttribute(
    std::string_view version_name, mlir::vhlo::Version version_id)
    : m_version_name(version_name) {
  m_version_id[0] = version_id.getMajor();
  m_version_id[1] = version_id.getMinor();
  m_version_id[2] = version_id.getPatch();
}

PJRT_NamedValue StableHLOVersionAttribute::toNamedValue() const {
  PJRT_NamedValue named_value;
  named_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  named_value.extension_start = nullptr;
  named_value.name = m_version_name.data();
  named_value.name_size = m_version_name.size();
  named_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
  named_value.int64_array_value = m_version_id;
  named_value.value_size = c_version_id_size;

  return named_value;
}

const std::vector<PJRT_NamedValue> &PluginAttributes::getAttributes() {
  static StableHLOVersionAttribute current{
      StableHLOVersionAttribute::current_version_attribute_name,
      mlir::vhlo::Version::getCurrentVersion()};
  static StableHLOVersionAttribute minimum{
      StableHLOVersionAttribute::minimum_version_attribute_name,
      mlir::vhlo::Version::getMinimumVersion()};

  static const std::vector<PJRT_NamedValue> attrs = []() {
    std::vector<PJRT_NamedValue> versions;
    versions.reserve(2);
    versions.push_back(current.toNamedValue());
    versions.push_back(minimum.toNamedValue());
    return versions;
  }();

  return attrs;
}

void PluginAttributes::bindApi(PJRT_Api *api) {
  api->PJRT_Plugin_Initialize = internal::onPluginInitialize;
  api->PJRT_Plugin_Attributes = internal::onPluginAttributes;
}

namespace internal {

PJRT_Error *onPluginInitialize(PJRT_Plugin_Initialize_Args *args) {
  DLOG_F(LOG_DEBUG, "PluginAttributes::PJRT_Plugin_Initialize");

  return nullptr;
}

PJRT_Error *onPluginAttributes(PJRT_Plugin_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "PluginAttributes::PJRT_Plugin_Attributes");

  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();
  args->attributes = attrs.data();
  args->num_attributes = attrs.size();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

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

PluginAttributes::PluginAttributes()
    : m_stablehlo_current_version("stablehlo_current_version",
                                  mlir::vhlo::Version::getCurrentVersion()),
      m_stablehlo_minimum_version("stablehlo_minimum_version",
                                  mlir::vhlo::Version::getMinimumVersion()) {
  m_attributes.emplace_back(m_stablehlo_current_version.toNamedValue());
  m_attributes.emplace_back(m_stablehlo_minimum_version.toNamedValue());
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

  static std::unique_ptr<PluginAttributes> s_plugin_attributes =
      std::make_unique<PluginAttributes>();
  args->attributes = s_plugin_attributes->getAttributes();
  args->num_attributes = s_plugin_attributes->getNumAttributes();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt

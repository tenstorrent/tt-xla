// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/plugin_attributes.h"

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

PJRTPluginAttributes::PJRTPluginAttributes()
    : m_stablehlo_current_version("stablehlo_current_version",
                                  mlir::vhlo::Version::getCurrentVersion()),
      m_stablehlo_minimum_version("stablehlo_minimum_version",
                                  mlir::vhlo::Version::getMinimumVersion()) {
  m_attributes.emplace_back(m_stablehlo_current_version.toNamedValue());
  m_attributes.emplace_back(m_stablehlo_minimum_version.toNamedValue());
}

} // namespace tt::pjrt

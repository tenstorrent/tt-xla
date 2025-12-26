// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_PLUGIN_ATTRIBUTES_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_PLUGIN_ATTRIBUTES_H_

// c++ standard library includes
#include <string_view>
#include <vector>

// stablehlo includes
#include "stablehlo/dialect/Version.h"

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

// Models StableHLO version attributes.
class StableHLOVersionAttribute {
public:
  StableHLOVersionAttribute(std::string_view version_name,
                            mlir::vhlo::Version version_id);

  PJRT_NamedValue toNamedValue() const;

  static constexpr std::string_view current_version_attribute_name =
      "stablehlo_current_version";
  static constexpr std::string_view minimum_version_attribute_name =
      "stablehlo_minimum_version";

private:
  static constexpr std::size_t c_version_id_size = 3;

  std::string_view m_version_name;

  std::int64_t m_version_id[c_version_id_size];
};

// Container for PJRT plugin attributes that frameworks can read.
class PluginAttributes {
public:
  // Binds PJRT API functions implementation related to PJRT_Plugin.
  static void bindApi(PJRT_Api *api);

  // Holds attributes for:
  // - the current StableHLO version:
  // If a PJRT plugin has an attribute for `stablehlo_current_version` then JAX
  // will precisely downgrade the IR to the plugin's version. Without the
  // attribute JAX uses 12 weeks IR downgrade, meaning newer features can't be
  // used or integrated for several months. Similarly an older than 12w plugin
  // will have more stability if it lets JAX know its precise version so it
  // downgrades more than 12w. Note that support >12w isn't guaranteed by JAX
  // but historically has been fairly stable.
  // - the minimum supported StableHLO version:
  // Requires frameworks to upgrade the IR to at least this version, and to not
  // downgrade the IR below this version.
  static const std::vector<PJRT_NamedValue> &getAttributes();

private:
  PluginAttributes() = delete;
};

namespace internal {

// Implements PJRT_Plugin_Initialize API function.
PJRT_Error *onPluginInitialize(PJRT_Plugin_Initialize_Args *args);

// Implements PJRT_Plugin_Attributes API function.
PJRT_Error *onPluginAttributes(PJRT_Plugin_Attributes_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_PLUGIN_ATTRIBUTES_H_

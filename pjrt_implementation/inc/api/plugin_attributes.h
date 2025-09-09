// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_SRC_COMMON_PLUGIN_ATTRIBUTES_H_
#define TT_XLA_SRC_COMMON_PLUGIN_ATTRIBUTES_H_

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

private:
  static constexpr std::size_t c_version_id_size = 3;

  std::string_view m_version_name;

  std::int64_t m_version_id[c_version_id_size];
};

// Container for PJRT plugin attributes that frameworks can read.
class PJRTPluginAttributes {
public:
  PJRTPluginAttributes();

  const PJRT_NamedValue *getAttributes() const { return m_attributes.data(); }

  std::size_t getNumAttributes() const { return m_attributes.size(); }

private:
  std::vector<PJRT_NamedValue> m_attributes;

  // Attribute for the current StableHLO version of the plugin.
  // If a PJRT plugin has an attribute for `stablehlo_current_version` then JAX
  // will precisely downgrade the IR to the plugin's version. Without the
  // attribute JAX uses 12 weeks IR downgrade, meaning newer features can't be
  // used or integrated for several months. Similarly an older than 12w plugin
  // will have more stability if it lets JAX know its precise version so it
  // downgrades more than 12w. Note that support >12w isn't guaranteed by JAX
  // but historically has been fairly stable.
  StableHLOVersionAttribute m_stablehlo_current_version;

  // Attribute for the minimum supported StableHLO version of the plugin.
  // Requires frameworks to upgrade the IR to at least this version, and to not
  // downgrade the IR below this version.
  StableHLOVersionAttribute m_stablehlo_minimum_version;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_PLUGIN_ATTRIBUTES_H_

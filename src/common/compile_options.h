// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_SRC_COMMON_COMPILE_OPTIONS_H_
#define TT_XLA_SRC_COMMON_COMPILE_OPTIONS_H_

#include <string>
#include <unordered_map>

namespace tt::pjrt {
// POD struct containing various options used to customize module compilation.
struct CompileOptions {
  // Enables the ttmlir optimizer, i.e. the optimization passes and memory
  // layout analysis.
  bool enable_optimizer = false;
  bool codegen_cpp = false;

  static CompileOptions
  parse(const std::unordered_map<std::string, std::string> &compile_options);
};

namespace internal {

// Parse out the value of one specific boolean flag from the options map.
bool parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_COMPILE_OPTIONS_H_

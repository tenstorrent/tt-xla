// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/pjrt_implementation/module_builder/compile_options.h"

// c++ standard library includes
#include <algorithm>

namespace tt::pjrt::module_builder {

CompileOptions CompileOptions::parse(
    const std::unordered_map<std::string, std::string> &compile_options) {
  CompileOptions options;

  options.enable_optimizer =
      internal::parseBoolOption(compile_options, "optimize");
  options.enable_bfp8_conversion =
      internal::parseBoolOption(compile_options, "bfp8");

  return options;
}

namespace internal {

bool parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name) {
  if (compile_options.find(option_name) != compile_options.end()) {
    std::string option_value = compile_options.at(option_name);
    std::transform(option_value.begin(), option_value.end(),
                   option_value.begin(), ::tolower);
    return option_value == "true" || option_value == "1" ||
           option_value == "yes" || option_value == "on";
  }
  return false;
}

} // namespace internal

} // namespace tt::pjrt::module_builder

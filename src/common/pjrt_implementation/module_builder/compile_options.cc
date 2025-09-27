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
      internal::parseBoolOption(compile_options, "enable_optimizer");
  options.enable_sharding =
      internal::parseBoolOption(compile_options, "enable_sharding");
  options.enable_l1_interleaved =
      internal::parseBoolOption(compile_options, "enable_l1_interleaved");
  options.enable_bfp8_conversion =
      internal::parseBoolOption(compile_options, "enable_bfp8_conversion");
  options.enable_fusing_conv2d_with_multiply_pattern =
      internal::parseBoolOption(compile_options,
                                "enable_fusing_conv2d_with_multiply_pattern");

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

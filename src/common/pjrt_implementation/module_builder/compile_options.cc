// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/pjrt_implementation/module_builder/compile_options.h"
#include "common/status.h"

// c++ standard library includes
#include <algorithm>
#include <string>

namespace tt::pjrt {

CompileOptions CompileOptions::parse(
    const std::unordered_map<std::string, std::string> &compile_options) {
  CompileOptions options;

  options.enable_optimizer =
      internal::parseBoolOption(compile_options, "enable_optimizer")
          .value_or(false);
  options.enable_bfp8_conversion =
      internal::parseBoolOption(compile_options, "enable_bfp8_conversion")
          .value_or(false);
  options.backend = internal::parseBackendOption(compile_options, "backend")
                        .value_or(Backend::Default);

  return options;
}

namespace internal {

std::optional<bool> parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name) {
  if (auto it = compile_options.find(option_name);
      it != compile_options.end()) {
    std::string option_value = it->second;
    std::transform(option_value.begin(), option_value.end(),
                   option_value.begin(), ::tolower);

    if (option_value == "true" || option_value == "1" ||
        option_value == "yes" || option_value == "on") {
      return true;
    } else if (option_value == "false" || option_value == "0" ||
               option_value == "no" || option_value == "off") {
      return false;
    }
    ABORT_F("Unknown boolean option value: %s for %s", option_value.c_str(),
            option_name.c_str());
  }
  return std::nullopt;
}

std::optional<Backend> parseBackendOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name) {
  if (auto it = compile_options.find(option_name);
      it != compile_options.end()) {
    std::string option_value = it->second;
    std::transform(option_value.begin(), option_value.end(),
                   option_value.begin(), ::tolower);
    if (option_value == "default") {
      return Backend::Default;
    } else if (option_value == "codegen_cpp") {
      return Backend::CodegenCpp;
    } else if (option_value == "codegen_py") {
      return Backend::CodegenPy;
    }
    ABORT_F("Unknown backend option value: %s for %s", option_value.c_str(),
            option_name.c_str());
  }
  return Backend::Default;
}

} // namespace internal

} // namespace tt::pjrt

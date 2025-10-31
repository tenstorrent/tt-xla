// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_options.h"
#include "utils/logging.h"

// c++ standard library includes
#include <algorithm>
#include <string>

namespace tt::pjrt {

CompileOptions CompileOptions::parse(
    const std::unordered_map<std::string, std::string> &compile_options) {
  CompileOptions options;

  options.optimization_level = internal::parseOptimizationLevelOption(
                                   compile_options, "optimization_level")
                                   .value_or(options.optimization_level);

  options.enable_bfp8_conversion =
      internal::parseBoolOption(compile_options, "enable_bfp8_conversion")
          .value_or(options.enable_bfp8_conversion);
  options.backend = internal::parseBackendOption(compile_options, "backend")
                        .value_or(options.backend);
  options.enable_trace =
      internal::parseBoolOption(compile_options, "enable_trace")
          .value_or(options.enable_trace);
  options.dump_inputs =
      internal::parseBoolOption(compile_options, "dump_inputs")
          .value_or(options.dump_inputs);
  options.enable_const_eval =
      internal::parseBoolOption(compile_options, "enable_const_eval")
          .value_or(true);
  options.export_path =
      internal::parseStringOption(compile_options, "export_path");
  if (!options.export_path.has_value() &&
      options.backend != BackendRuntime::TTNNFlatbuffer) {
    ABORT_F("Compile option 'export_path' must be provided when backend is not "
            "'TTNNFlatbuffer'");
  }

  return options;
}

namespace internal {

std::optional<bool> parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name) {
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

std::optional<BackendRuntime> parseBackendOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name) {
  if (auto it = compile_options.find(option_name);
      it != compile_options.end()) {
    std::string option_value = it->second;
    std::transform(option_value.begin(), option_value.end(),
                   option_value.begin(), ::tolower);
    if (option_value == "default" || option_value == "ttnn_flatbuffer") {
      return BackendRuntime::TTNNFlatbuffer;
    } else if (option_value == "codegen_cpp") {
      return BackendRuntime::TTNNCodegenCpp;
    } else if (option_value == "codegen_py") {
      return BackendRuntime::TTNNCodegenPy;
    }
    ABORT_F("Unknown backend option value: %s for %s", option_value.c_str(),
            option_name.c_str());
  }
  return BackendRuntime::TTNNFlatbuffer;
}

std::optional<std::string> parseStringOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name) {
  auto it = compile_options.find(option_name);

  return it == compile_options.end() ? std::nullopt : std::optional(it->second);
}

std::optional<int> parseOptimizationLevelOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name) {
  static constexpr int max_optimization_level = 2;
  if (auto it = compile_options.find(option_name);
      it != compile_options.end()) {
    try {
      int level = std::stoi(it->second);
      if (level < 0 || level > 2) {
        ABORT_F("Invalid optimization_level: %d. Must be between 0 and %d.",
                level, max_optimization_level);
      }
      return level;
    } catch (const std::exception &e) {
      ABORT_F("Failed to parse optimization_level: %s. Must be an integer "
              "between 0 and %d ",
              e.what(), max_optimization_level);
    }
  }
  return std::nullopt;
}

bool shouldEnableOptimizer(int optimization_level) {
  return optimization_level >= 1;
}

bool shouldEnableMemoryLayoutAnalysis(int optimization_level) {
  return optimization_level >= 2;
}

bool shouldEnableFusingConv2dWithMultiplyPattern(int optimization_level) {
  return optimization_level >= 1;
}

} // namespace internal

} // namespace tt::pjrt

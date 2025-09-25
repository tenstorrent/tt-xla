// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_COMPILE_OPTIONS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_COMPILE_OPTIONS_H_

// c++ standard library includes
#include <optional>
#include <string>
#include <unordered_map>

namespace tt::pjrt {

// Enumeration for backend types
enum class Backend {
  Default,    // "default" - standard TT hardware targeting
  CodegenCpp, // "codegen_cpp" - TTNN C++ code generation
  CodegenPy   // "codegen_py" - TTNN Python code generation
};

// POD struct containing various options used to customize module compilation.
struct CompileOptions {
  // Enables the ttmlir optimizer, i.e. the optimization passes and memory
  // layout analysis.
  bool enable_optimizer = false;

  // Enables automatic MLIR graph conversion into block fp8 format. This is
  // supported only when the graph is in bfloat16 format, to avoid loss in
  // precision. Final graph will have input and output nodes in bfloat16 and
  // everything else in bfp8. Essentially adding type casts at the beginning and
  // in the end of the graph, while all intermediate results are in bfp8. This
  // bfloat16 wrapping is done because block formats are TT hardware specific,
  // and user should provide and get tensors of common dtype.
  bool enable_bfp8_conversion = false;

  // We offer the option to use TTNN code generation as an alternate way to
  // target the hardware. This is also sometimes reffered to as EmitC/EmitPy.
  // Valid values: Backend::Default, Backend::CodegenCpp, Backend::CodegenPy
  Backend backend = Backend::Default;

  // Enables saving graph inputs to disk whenever Execute() is called.
  // This is useful for chisel and codegen. Defaults to false on the default
  // backend and true on the codegen backend.
  std::optional<bool> dump_inputs = std::nullopt;

  // Path that will contain the codegen solution and saved inputs.
  std::string export_path = "";

  static CompileOptions
  parse(const std::unordered_map<std::string, std::string> &compile_options);
};

namespace internal {

// Parse out the value of one specific boolean flag from the options map.
std::optional<bool> parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);

// Parse backend option from string to enum
std::optional<Backend> parseBackendOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);

std::optional<std::string> parseStringOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);

} // namespace internal

// Utility functions for Backend enum
const char *backendToString(Backend backend);
Backend stringToBackend(const std::string &str);

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_COMPILE_OPTIONS_H_

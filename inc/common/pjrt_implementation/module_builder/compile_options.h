// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// c++ standard library includes
#include <string>
#include <unordered_map>

namespace tt::pjrt::module_builder {

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

  static CompileOptions
  parse(const std::unordered_map<std::string, std::string> &compile_options);
};

namespace internal {

// Parse out the value of one specific boolean flag from the options map.
bool parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);

} // namespace internal

} // namespace tt::pjrt::module_builder

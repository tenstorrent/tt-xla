// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>

namespace tt::pjrt {
// POD struct containing various options used to customize module compilation.
struct CompileOptions {
  // Enables the ttmlir optimizer, i.e. the optimization passes and memory
  // layout analysis.
  bool enable_optimizer = false;

  static CompileOptions
  parse(const std::unordered_map<std::string, std::string> &compile_options);
};

namespace internal {
bool parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    std::string option_name);
}
} // namespace tt::pjrt

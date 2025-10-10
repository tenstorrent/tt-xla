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
enum class BackendRuntime {
  // Targets tt-mlir TTNN runtime to execute the compiled flatbuffer.
  TTNNFlatbuffer,

  // Generates TTNN C++ code to be compiled and executed.
  TTNNCodegenCpp,

  // Generates TTNN Python code.
  TTNNCodegenPy
};

// POD struct containing various options used to customize module compilation.
struct CompileOptions {
  // Enables optimizer passes in MLIR. This includes various optimizations
  // such as improving tensor memory layouts, operation configurations etc.
  bool enable_optimizer = false;

  // Enables memory layout analysis to allow sharded memory layouts in optimizer
  // passes.
  bool enable_memory_layout_analysis = false;

  // Enables L1 interleaved fallback analysis in optimizer passes.
  // This analysis attempts to move tensors from DRAM to L1 memory with
  // interleaved layout when beneficial for performance.
  bool enable_l1_interleaved = false;

  // Enables automatic MLIR graph conversion into block fp8 format. This is
  // supported only when the graph is in bfloat16 format, to avoid loss in
  // precision. Final graph will have input and output nodes in bfloat16 and
  // everything else in bfp8. Essentially adding type casts at the beginning and
  // in the end of the graph, while all intermediate results are in bfp8. This
  // bfloat16 wrapping is done because block formats are TT hardware specific,
  // and user should provide and get tensors of common dtype.
  bool enable_bfp8_conversion = false;

  // Enables Conv2d fusion with multiply pattern in the TTNN fusing pass.
  // TODO(sdjordjevicTT): This is a temporary option and will be removed once
  // the underlying issue https://github.com/tenstorrent/tt-mlir/issues/4628 is
  // fixed.
  bool enable_fusing_conv2d_with_multiply_pattern = false;

  // Backend runtime which should be targeted for compilation and execution.
  BackendRuntime backend = BackendRuntime::TTNNFlatbuffer;

  // Enables trace hoisting for TTNN pipeline.
  // This is supported only when all non-consteval ops are on device.
  // This is a performance optimization feature that will eliminate
  // host overhead for creating and dispatching operations
  // that are repeated multiple times.
  bool enable_trace = false;

  // Enables saving graph inputs to disk whenever Execute() is called.
  // This is useful for chisel and codegen.
  bool dump_inputs = false;

  // Path that will contain the codegen solution and saved inputs.
  std::optional<std::string> export_path = std::nullopt;

  static CompileOptions
  parse(const std::unordered_map<std::string, std::string> &compile_options);
};

namespace internal {

// Parse out the value of one specific boolean flag from the options map.
std::optional<bool> parseBoolOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name);

// Parse backend option from string to enum
std::optional<BackendRuntime> parseBackendOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name);

std::optional<std::string> parseStringOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_COMPILE_OPTIONS_H_

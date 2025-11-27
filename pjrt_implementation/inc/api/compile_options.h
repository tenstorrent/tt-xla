// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_H_

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
  // Optimization level (0, 1, or 2) that controls multiple optimization passes.
  // See documentation for details on what each level enables.
  // Level 0 (default): All optimizations disabled
  // Level 1: Basic optimizations (optimizer + Conv2d fusion)
  // Level 2: Advanced optimizations (optimizer + memory layout + Conv2d fusion)
  int optimization_level = 0;

  // Enables automatic MLIR graph conversion into block fp8 format. This is
  // supported only when the graph is in bfloat16 format, to avoid loss in
  // precision. Final graph will have input and output nodes in bfloat16 and
  // everything else in bfp8. Essentially adding type casts at the beginning and
  // in the end of the graph, while all intermediate results are in bfp8. This
  // bfloat16 wrapping is done because block formats are TT hardware specific,
  // and user should provide and get tensors of common dtype.
  bool enable_bfp8_conversion = false;

  // Enables experimental BFP8 weight conversion in MLIR.
  bool experimental_enable_weight_bfp8_conversion = false;

  // Enables Conv2d fusion with multiply pattern in the TTNN fusing pass.
  // TODO(sdjordjevicTT): This is a temporary option and will be removed once
  // the underlying issue https://github.com/tenstorrent/tt-mlir/issues/4628 is
  // fixed.
  bool experimental_enable_fusing_conv2d_with_multiply_pattern = false;

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
  bool export_tensors = false;

  // Enables generation of consteval graphs.
  //
  // We allow the user of the plugin to toggle consteval in tt-mlir. We would
  // like for this to be on at all times as it results in a more performant
  // model. However, the results of the consteval graphs are stored on device
  // until the device is closed. If multiple graphs use the same weights (i.e,
  // same model, different input shapes), then we will end up cloning the
  // weights multiple times. This can easily lead to OOM errors. There is an
  // issue tracking this in tt-mlir:
  // https://github.com/tenstorrent/tt-mlir/issues/3888
  bool enable_const_eval = true;

  // Enable collection of TTNN performance metrics during execution.
  bool ttnn_perf_metrics_enabled = false;

  // Output file path for TTNN performance metrics.
  // If empty, metrics will be saved to the "perf_metrics" directory with a
  // default name.
  std::string ttnn_perf_metrics_output_file = "";

  // Path that will contain any exported artifacts.
  // This includes: codegen solutions, graph inputs and intermediate IRs.
  // Setting this will enable IR dumping.
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

std::optional<int> parseIntOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_H_

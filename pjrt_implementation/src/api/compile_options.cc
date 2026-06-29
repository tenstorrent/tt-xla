// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_options.h"
#include "utils/logging.h"

// c++ standard library includes
#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace tt::pjrt {

CompileOptions CompileOptions::parse(
    const std::unordered_map<std::string, std::string> &compile_options) {
  CompileOptions options;

  options.optimization_level =
      internal::parseIntOption(compile_options, optimization_level_key)
          .value_or(options.optimization_level);
  options.experimental_weight_dtype =
      internal::parseStringOption(compile_options, "experimental_weight_dtype")
          .value_or(options.experimental_weight_dtype);
  options.experimental_kv_cache_dtype = internal::parseStringOption(
      compile_options, "experimental-kv-cache-dtype");
  options.math_fidelity =
      internal::parseStringOption(compile_options, "math_fidelity");

  options.fp32_dest_acc_en =
      internal::parseBoolOption(compile_options, "fp32_dest_acc_en");
  options.experimental_enable_fusing_conv2d_with_multiply_pattern =
      internal::parseBoolOption(
          compile_options,
          "experimental_enable_fusing_conv2d_with_multiply_pattern")
          .value_or(
              options.experimental_enable_fusing_conv2d_with_multiply_pattern);
  options.backend = internal::parseBackendOption(compile_options, "backend")
                        .value_or(options.backend);
  options.export_path =
      internal::parseStringOption(compile_options, "export_path");

  // The codegen emit/load env vars are a surface over the codegen backends:
  // they let a workflow opt in externally (e.g. flipping emit/load on a running
  // vLLM server without editing config code). Resolve them here -- right after
  // the explicit backend/export_path and BEFORE any backend-dependent default
  // below -- so those defaults all see the final backend. An explicit `backend`
  // always wins; an explicit export_path is overriden.
  const char *emit_dir = std::getenv("TTXLA_CODEGEN_EXPORT_DIR");
  const char *load_dir = std::getenv("TTXLA_CODEGEN_LOAD_DIR");
  if (!compile_options.count("backend")) {
    if (load_dir) {
      options.backend = BackendRuntime::TTNNCodegenLoadPy;
      options.export_path = load_dir;
    } else if (emit_dir) {
      options.backend = BackendRuntime::TTNNCodegenPy;
      // Emit a runnable forward() entrypoint so the code can be reloaded later.
      options.target_module = true;
      options.export_path = emit_dir;
    }
  }

  options.enable_trace =
      internal::parseBoolOption(compile_options, "enable_trace")
          .value_or(options.enable_trace);
  // By default, export tensors for codegen paths.
  options.export_tensors =
      internal::parseBoolOption(compile_options, "export_tensors")
          .value_or(options.backend == BackendRuntime::TTNNCodegenPy ||
                    options.backend == BackendRuntime::TTNNCodegenCpp);
  options.enable_const_eval =
      internal::parseBoolOption(compile_options, "enable_const_eval")
          .value_or(options.enable_const_eval);
  options.enable_const_eval_on_cpu =
      internal::parseBoolOption(compile_options, "enable_const_eval_on_cpu")
          .value_or(options.enable_const_eval_on_cpu);
  options.enable_const_eval_inputs_to_system_memory =
      internal::parseBoolOption(compile_options,
                                "enable_const_eval_inputs_to_system_memory")
          .value_or(options.enable_const_eval_inputs_to_system_memory);
  options.experimental_enable_permute_matmul_fusion =
      internal::parseBoolOption(compile_options,
                                "experimental_enable_permute_matmul_fusion")
          .value_or(options.experimental_enable_permute_matmul_fusion);
  options.codegen_try_recover_structure =
      internal::parseBoolOption(compile_options,
                                "codegen_try_recover_structure")
          .value_or(options.codegen_try_recover_structure);
  options.codegen_split_files =
      internal::parseBoolOption(compile_options, "codegen_split_files")
          .value_or(options.codegen_split_files);
  options.experimental_enable_dram_space_saving_optimization =
      internal::parseBoolOption(
          compile_options, "experimental-enable-dram-space-saving-optimization")
          .value_or(options.experimental_enable_dram_space_saving_optimization);
  options.enable_create_d2m_subgraphs =
      internal::parseBoolOption(compile_options, "enable_create_d2m_subgraphs")
          .value_or(options.enable_create_d2m_subgraphs);
  options.ttnn_perf_metrics_enabled =
      internal::parseBoolOption(compile_options, "ttnn_perf_metrics_enabled")
          .value_or(false);
  options.all_reduce_workaround_enabled =
      internal::parseBoolOption(compile_options,
                                "all_reduce_workaround_enabled")
          .value_or(options.all_reduce_workaround_enabled);
  options.ttnn_perf_metrics_output_file =
      internal::parseStringOption(compile_options,
                                  "ttnn_perf_metrics_output_file")
          .value_or("");
  // Default value of dry_run is dependent on backend
  bool is_default_dry_run = true;
  if (options.backend == BackendRuntime::TTNNFlatbuffer ||
      options.backend == BackendRuntime::TTNNCodegenLoadPy) {
    is_default_dry_run = false;
  }
  options.dry_run = internal::parseBoolOption(compile_options, "dry_run")
                        .value_or(is_default_dry_run);
  options.target_module =
      internal::parseBoolOption(compile_options, "target_module")
          .value_or(options.target_module);
  options.export_model_name =
      internal::parseStringOption(compile_options, "export_model_name")
          .value_or("");
  // Codegen lays out one graph_N directory per graph and matches graphs on load
  // by hash, so a model label has no role there and would only add a confusing
  // prefix to the emitted files. Clear it for every codegen backend; only the
  // IR-export (flatbuffer) path uses export_model_name.
  if (options.backend == BackendRuntime::TTNNCodegenPy ||
      options.backend == BackendRuntime::TTNNCodegenCpp ||
      options.backend == BackendRuntime::TTNNCodegenLoadPy) {
    options.export_model_name = "";
  }

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
    } else if (option_value == "codegen_load_py") {
      return BackendRuntime::TTNNCodegenLoadPy;
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

std::optional<int> parseIntOption(
    const std::unordered_map<std::string, std::string> &compile_options,
    const std::string &option_name) {
  static constexpr int max_optimization_level = 2;
  if (auto it = compile_options.find(option_name);
      it != compile_options.end()) {
    try {
      return std::stoi(it->second);
    } catch (const std::exception &e) {
      ABORT_F("Failed to parse optimization_level: %s. Must be an integer.",
              e.what());
    }
  }
  return std::nullopt;
}

} // namespace internal

} // namespace tt::pjrt

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/executable_image.h"

// c++ standard library includes
#include <functional>
#include <sstream>
#include <string>

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/memory_instance.h"

namespace tt::pjrt {

std::shared_ptr<ExecutableImage> ExecutableImage::createInstance(
    const tt::runtime::Binary &flatbuffer_binary,
    std::string original_mlir_code, std::string ttir_mlir_code,
    std::string ttnn_mlir_code, std::string executable_name,
    size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
    const std::vector<std::uint32_t> &devices_mesh_shape,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_sharding,
    const std::vector<bool> &is_output_scalar,
    const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
    module_builder::CompileOptions &&compile_options) {
  struct make_shared_enabler : public ExecutableImage {
    make_shared_enabler(
        const tt::runtime::Binary &flatbuffer_binary,
        std::string &&original_mlir_code, std::string &&ttir_mlir_code,
        std::string &&ttnn_mlir_code, std::string &&executable_name,
        size_t num_partitions, size_t num_replicas,
        size_t num_devices_to_utilize,
        const std::vector<std::uint32_t> &devices_mesh_shape,
        const std::vector<mlir::tt::sharding_utils::MeshSharding>
            &input_sharding,
        const std::vector<mlir::tt::sharding_utils::MeshSharding>
            &output_sharding,
        const std::vector<bool> &is_output_scalar,
        const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
        module_builder::CompileOptions &&compile_options)
        : ExecutableImage(flatbuffer_binary, std::move(original_mlir_code),
                          std::move(ttir_mlir_code), std::move(ttnn_mlir_code),
                          std::move(executable_name), num_partitions,
                          num_replicas, num_devices_to_utilize,
                          devices_mesh_shape, input_sharding, output_sharding,
                          is_output_scalar, expected_output_data_types,
                          std::move(compile_options)) {}
  };

  return std::make_shared<make_shared_enabler>(
      flatbuffer_binary, std::move(original_mlir_code),
      std::move(ttir_mlir_code), std::move(ttnn_mlir_code),
      std::move(executable_name), num_partitions, num_replicas,
      num_devices_to_utilize, devices_mesh_shape, input_sharding,
      output_sharding, is_output_scalar, expected_output_data_types,
      std::move(compile_options));
}

ExecutableImage::ExecutableImage(
    const tt::runtime::Binary &flatbuffer_binary,
    std::string &&original_mlir_code, std::string &&ttir_mlir_code,
    std::string &&ttnn_mlir_code, std::string &&executable_name,
    size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
    const std::vector<std::uint32_t> &devices_mesh_shape,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_sharding,
    const std::vector<bool> &is_output_scalar,
    const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
    module_builder::CompileOptions &&compile_options)
    : m_flatbuffer_binary(flatbuffer_binary),
      m_original_mlir_code(std::move(original_mlir_code)),
      m_ttir_mlir(std::move(ttir_mlir_code)),
      m_ttnn_mlir(std::move(ttnn_mlir_code)),
      m_executable_name(std::move(executable_name)),
      m_num_partitions(num_partitions), m_num_replicas(num_replicas),
      m_num_devices_to_utilize(num_devices_to_utilize),
      m_devices_mesh_shape(devices_mesh_shape),
      m_input_sharding(input_sharding), m_output_sharding(output_sharding),
      m_output_types(expected_output_data_types),
      m_compile_options(std::move(compile_options)) {

  // Generate fingerprint after all dependencies are initialized
  m_fingerprint = generateFingerprint();

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  m_num_inputs = m_flatbuffer_binary.getProgramInputs(program_index).size();
  std::vector<tt::runtime::TensorDesc> output_specs =
      m_flatbuffer_binary.getProgramOutputs(program_index);
  m_num_outputs = output_specs.size();

  // We expect that these conditions are satisfied and checked in module builder
  // so we just assert here.
  assert(m_num_inputs == input_sharding.size());
  assert(m_num_outputs == output_sharding.size());
  assert(m_num_outputs == is_output_scalar.size());

  m_output_types.resize(m_num_outputs);
  m_output_dimensions.reserve(m_num_outputs);
  m_output_ranks.resize(m_num_outputs);

  for (size_t output_index = 0; output_index < m_num_outputs; ++output_index) {
    // PJRT expects an empty shape for scalars.
    m_output_dimensions.emplace_back(is_output_scalar[output_index]
                                         ? std::vector<std::uint32_t>()
                                         : output_specs[output_index].shape);

    m_output_ranks[output_index] = m_output_dimensions[output_index].size();

    for (std::uint32_t dim : m_output_dimensions[output_index]) {
      m_output_dimensions_flat.push_back(static_cast<std::int64_t>(dim));
    }
  }

  m_output_memory_kinds.reserve(m_num_outputs);
  m_output_memory_kinds_sizes.reserve(m_num_outputs);

  // Currently we move all output buffers to host memory at the end of
  // PJRT_LoadedExecutable_Execute.
  for (size_t output_index = 0; output_index < m_num_outputs; ++output_index) {
    m_output_memory_kinds.emplace_back(
        MemoryInstance::c_device_memory_kind_name.c_str());
    m_output_memory_kinds_sizes.emplace_back(
        MemoryInstance::c_device_memory_kind_name.size());
  }
}

const std::vector<std::uint32_t> &
ExecutableImage::getOutputShape(size_t output_index) const {
  assert(output_index < m_output_dimensions.size() &&
         "Output index out of range");
  return m_output_dimensions[output_index];
}

const mlir::tt::sharding_utils::MeshSharding &
ExecutableImage::getInputSharding(size_t input_index) const {
  assert(input_index < m_input_sharding.size() && "Input index out of range");
  return m_input_sharding[input_index];
}

const mlir::tt::sharding_utils::MeshSharding &
ExecutableImage::getOutputSharding(size_t output_index) const {
  assert(output_index < m_output_sharding.size() &&
         "Output index out of range");
  return m_output_sharding[output_index];
}

std::string ExecutableImage::generateFingerprint() const {
  std::stringstream data_to_hash;

  // 1. Add MLIR code
  data_to_hash << "mlir:" << m_original_mlir_code << "\n";

  // 2. Add compile options
  data_to_hash << "enable_optimizer:" << m_compile_options.enable_optimizer
               << "\n";
  data_to_hash << "enable_memory_layout_analysis:"
               << m_compile_options.enable_memory_layout_analysis << "\n";
  data_to_hash << "enable_l1_interleaved:"
               << m_compile_options.enable_l1_interleaved << "\n";
  data_to_hash << "enable_bfp8_conversion:"
               << m_compile_options.enable_bfp8_conversion << "\n";
  data_to_hash << "enable_fusing_conv2d_with_multiply_pattern:"
               << m_compile_options.enable_fusing_conv2d_with_multiply_pattern
               << "\n";

  // 3. Add compiler version
  data_to_hash << "ttmlir_version:" << m_flatbuffer_binary.getVersion() << "\n";

  // 4. Generate hash using std::hash
  std::hash<std::string> hasher;
  size_t hash_value = hasher(data_to_hash.str());

  // Convert to hex string
  std::stringstream hex_ss;
  hex_ss << std::hex << hash_value;
  return hex_ss.str();
}

} // namespace tt::pjrt

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

#include "common/pjrt_implementation/data_type_utils.h"

namespace tt::pjrt {

std::shared_ptr<ExecutableImage> ExecutableImage::createInstance(
    const tt::runtime::Binary &flatbuffer_binary,
    std::string &&optimized_mlir_code, std::string &&executable_name,
    size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_sharding,
    const std::vector<bool> &is_output_scalar) {
  return std::make_shared<ExecutableImage>(
      flatbuffer_binary, std::move(optimized_mlir_code),
      std::move(executable_name), num_partitions, num_replicas,
      num_devices_to_utilize, input_sharding, output_sharding,
      is_output_scalar);
}

ExecutableImage::ExecutableImage(
    const tt::runtime::Binary &flatbuffer_binary,
    std::string &&optimized_mlir_code, std::string &&executable_name,
    size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_sharding,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::vector<bool> &is_output_scalar)
    : m_flatbuffer_binary(flatbuffer_binary),
      m_optimized_mlir_code(std::move(optimized_mlir_code)),
      m_executable_name(std::move(executable_name)),
      m_num_partitions(num_partitions), m_num_replicas(num_replicas),
      m_num_devices_to_utilize(num_devices_to_utilize),
      m_input_sharding(input_sharding), m_output_sharding(output_sharding) {

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  m_num_inputs = m_binary.getProgramInputs(program_index).size();
  std::vector<tt::runtime::TensorDesc> output_specs =
      m_binary.getProgramOutputs(program_index);
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
    m_output_types[output_index] =
        tt::pjrt::data_type_utils::convertRuntimeToPJRTDataType(
            output_specs[output_index].dataType);

    // PJRT expects an empty shape for scalars.
    m_output_dimensions.emplace_back(is_output_scalar[i]
                                         ? std::vector<std::uint32_t>()
                                         : output_specs[i].shape);

    m_output_ranks[output_index] = m_output_dimensions[output_index].size();

    for (std::uint32_t dim : m_output_dimensions[output_index]) {
      m_output_dimensions_flat.push_back(static_cast<std::int64_t>(dim));
    }
  }
}

const std::vector<std::uint32_t> &
ExecutableImage::getOutputShape(size_t output_index) const {
  assert(index < m_output_dims.size() && "Output index out of range");
  return m_output_dims[index];
}

const mlir::tt::sharding_utils::MeshSharding &
ExecutableImage::getInputSharding(size_t input_index) const {
  assert(input_index < m_input_sharding.size() && "Input index out of range");
  return m_input_sharding[input_index];
}

const mlir::tt::sharding_utils::MeshSharding &
getOutputSharding(size_t output_index) const {
  assert(output_index < m_output_sharding.size() &&
         "Output index out of range");
  return m_output_sharding[output_index];
}

} // namespace tt::pjrt

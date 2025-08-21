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
#include <cassert>
#include <functional>
#include <sstream>
#include <string>

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/memory_instance.h"

namespace tt::pjrt {

ExecutableImage::ExecutableImage() : m_flatbuffer_binary(nullptr) {}

bool vectors_same(std::vector<std::uint32_t> a, std::vector<std::uint32_t> b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

void ExecutableImage::validate() {
  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  assert(m_num_inputs ==
         m_flatbuffer_binary.getProgramInputs(program_index).size());
  std::vector<tt::runtime::TensorDesc> output_specs =
      m_flatbuffer_binary.getProgramOutputs(program_index);
  assert(m_num_outputs == output_specs.size());

  assert(m_num_inputs == m_input_sharding.size());
  assert(m_num_outputs == m_output_sharding.size());

  int outputs_so_far = 0;
  for (size_t output_index = 0; output_index < m_num_outputs; ++output_index) {
    assert(vectors_same(m_output_dimensions[output_index],
                        output_specs[output_index].shape) &&
           "Output shape from flatbuffer binary does not match the one "
           "collected from the MLIR module");

    assert(m_output_ranks[output_index] ==
               m_output_dimensions[output_index].size() &&
           "Output rank from flatbuffer binary does not match the one "
           "collected from the MLIR module");

    for (auto dim_index = 0;
         dim_index < m_output_dimensions[output_index].size(); dim_index++) {
      assert(m_output_dimensions_flat[outputs_so_far + dim_index] ==
                 static_cast<std::int64_t>(
                     m_output_dimensions[output_index][dim_index]) &&
             "Output flat dimension from flatbuffer binary does not match the "
             "one collected from the MLIR module");
    }

    outputs_so_far += m_output_dimensions[output_index].size();
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
  data_to_hash << "enable_bfp8_conversion:"
               << m_compile_options.enable_bfp8_conversion << "\n";

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

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
#include <memory>

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"

namespace tt::pjrt {

const std::string_view kMlirFormat = "mlir";

ExecutableImage::ExecutableImage(
    const tt::runtime::Binary &binary, const std::string &code,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_sharding,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::vector<bool> &is_output_scalar)
    : m_ref_count(1), m_binary(binary), m_code(code),
      m_input_sharding(input_sharding), m_output_sharding(output_sharding) {

  std::vector<tt::runtime::TensorDesc> output_specs =
      m_binary.getProgramOutputs(0);
  m_result_count = output_specs.size();
  m_arg_count = m_binary.getProgramInputs(0).size();

  if (m_result_count != is_output_scalar.size()) {
    // TODO: Move to module_builder and just assert here
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %zu "
           "than expected %zu",
           m_result_count, is_output_scalar.size());
  }

  m_output_types.resize(m_result_count);
  m_output_dims.resize(m_result_count);
  m_output_strides.resize(m_result_count);
  for (int i = 0; i < m_result_count; i++) {
    m_output_types[i] = tt::pjrt::data_type_utils::convertRuntimeToPJRTDataType(
        output_specs[i].dataType);

    // PJRT expects an empty shape for scalars.
    m_output_dims[i] = is_output_scalar[i] ? std::vector<std::uint32_t>()
                                           : output_specs[i].shape;

    m_output_strides[i] = output_specs[i].stride;
  }
}

const std::vector<std::uint32_t> &
ExecutableImage::get_output_shape(const size_t index) const {
  assert(index < m_output_dims.size() && "Output index out of range");
  return m_output_dims[index];
}

const std::vector<std::uint32_t> &
ExecutableImage::get_output_stride(const size_t index) const {
  assert(index < m_output_strides.size() && "Output index out of range");
  return m_output_strides[index];
}

void ExecutableImage::populateOutputDimsConcatenated() {
  size_t num_dims = 0;

  m_output_dim_sizes = std::make_unique<size_t[]>(m_result_count);
  for (size_t i = 0; i < m_result_count; i++) {
    m_output_dim_sizes[i] = m_output_dims[i].size();
    num_dims += m_output_dim_sizes[i];
  }

  m_output_dims_concatenated = std::make_unique<int64_t[]>(num_dims);
  size_t dims_index = 0;
  for (size_t i = 0; i < m_result_count; i++) {
    for (size_t j = 0; j < m_output_dim_sizes[i]; j++) {
      m_output_dims_concatenated[dims_index + j] = m_output_dims[i][j];
    }
    dims_index += m_output_dim_sizes[i];
  }
}

void ExecutableImage::get_output_dims_concatenated(const size_t **dim_sizes,
                                                   const int64_t **dims) {
  if (!areOutputDimsConcatenated()) {
    populateOutputDimsConcatenated();
  }

  *dim_sizes = m_output_dim_sizes.get();
  *dims = m_output_dims_concatenated.get();
}

} // namespace tt::pjrt

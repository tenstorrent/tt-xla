// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include <atomic>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

class ExecutableImage {

public:
  ExecutableImage(
      const tt::runtime::Binary &binary, std::string code,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::vector<bool> &is_output_scalar);

  operator PJRT_Executable *() {
    return reinterpret_cast<PJRT_Executable *>(this);
  }
  static ExecutableImage *Unwrap(PJRT_Executable *exe) {
    return reinterpret_cast<ExecutableImage *>(exe);
  }
  static void BindApi(PJRT_Api *api);

  void AddRef() { m_ref_count.fetch_add(1); }
  void DecRef() {
    if (m_ref_count.fetch_sub(1) == 0) {
      delete this;
    }
  }

  const size_t get_arg_count() const { return m_arg_count; }

  const size_t get_result_count() const { return m_result_count; }

  const tt::runtime::Binary &get_binary() const { return m_binary; }

  const std::string &get_code() const { return m_code; }

  const std::vector<std::uint32_t> &get_output_shape(const size_t index) const;

  const std::vector<std::uint32_t> &get_output_stride(const size_t index) const;

  PJRT_Buffer_Type *get_output_types() { return m_output_types.data(); }

  size_t get_num_outputs() const { return m_output_types.size(); }

  const std::vector<std::uint32_t> &get_mesh_shape() const {
    return m_mesh_shape;
  }

  // Checks if the output on the i-th index is a scalar.
  bool isOutputScalar(size_t index) const;

  // Returns the sharding information for the i-th input.
  const mlir::tt::sharding_utils::MeshSharding &
  getInputSharding(size_t index) const {
    return m_input_sharding[index];
  }

  // Returns the sharding information for the i-th output.
  const mlir::tt::sharding_utils::MeshSharding &
  getOutputSharding(size_t index) const {
    return m_output_sharding[index];
  }

private:
  // Retrieves pointers to the concatenated list of output dimensions and the
  // corresponding list of ranks (number of dimensions per output tensor).
  void get_output_dims_concatenated(const size_t **dim_sizes,
                                    const int64_t **dims);

  // Checks if the concatenated output dimensions and the corresponding rank
  // list have been initialized.
  bool areOutputDimsConcatenated() const {
    return m_output_dim_sizes && m_output_dims_concatenated;
  }

  // Populates the concatenated list of output dimensions and the corresponding
  // list of ranks.
  void populateOutputDimsConcatenated();

  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> m_ref_count;

  // Raw compiler output.
  tt::runtime::Binary m_binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string m_code;

  size_t m_arg_count;
  size_t m_result_count;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;

  // For every output, holds PJRT_Buffer_Type.
  std::vector<PJRT_Buffer_Type> m_output_types;

  // For every output, holds a list of its dimensions.
  std::vector<std::vector<uint32_t>> m_output_dims;

  // For every output, stores rank (number of dimensions). Nullptr until its
  // getter is called.
  std::unique_ptr<size_t[]> m_output_dim_sizes;

  // Stores all output dimensions concatenated in a flat array. Nullptr until
  // its getter is called.
  std::unique_ptr<int64_t[]> m_output_dims_concatenated;

  // For every output, holds its stride.
  std::vector<std::vector<uint32_t>> m_output_strides;

  // Hold the sharding information for each input.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> m_input_sharding;

  // Hold the sharding information for each output.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> m_output_sharding;

  // Device mesh shape required by this image.
  const std::vector<std::uint32_t> m_mesh_shape;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

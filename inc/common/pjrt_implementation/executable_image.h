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
#include "tt/runtime/types.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

class ExecutableImage {

public:
  ExecutableImage(const tt::runtime::Binary &binary, std::string code,
                  const std::vector<bool> &is_output_scalar,
                  size_t num_addressable_devices);

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

  const size_t get_num_addressable_devices() const {
    return m_num_addressable_devices;
  }

  const std::vector<std::uint32_t> &get_output_shape(size_t index) const;

  void get_output_dims_concatenated(const size_t **dim_sizes,
                                    const int64_t **dims);

  PJRT_Buffer_Type *get_output_types() { return m_output_types.data(); }
  size_t num_output_types() { return m_output_types.size(); }

private:
  // Checks whether m_output_dim_sizes and m_output_dims_concatenated have been
  // filled.
  bool areOutputDimsConcatinated() const {
    return m_output_dim_sizes && m_output_dims_concatenated;
  }

  // Fills the m_output_dim_sizes and m_output_dims_concatenated array.
  void populateOutputDimsConcatenated();

  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> m_ref_count;

  // Raw compiler output.
  tt::runtime::Binary m_binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string m_code;

  size_t m_arg_count;
  size_t m_result_count;
  size_t m_num_addressable_devices;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;

  // For every output, holds PJRT_Buffer_Type.
  std::vector<PJRT_Buffer_Type> m_output_types;

  // For every output, holds a list of its dimensions.
  std::vector<std::vector<uint32_t>> m_output_dims;

  // For every output, holds how many dimensions it has. Nullptr until getter
  // get_output_dims_concatenated is called.
  std::unique_ptr<size_t[]> m_output_dim_sizes;

  // Holds all output dimensions concatenated. Nullptr until getter
  // get_output_dims_concatenated is called.
  std::unique_ptr<int64_t[]> m_output_dims_concatenated;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

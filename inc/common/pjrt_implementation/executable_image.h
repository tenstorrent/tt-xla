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
#include <memory>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/types.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

class ExecutableImage {

public:
  ExecutableImage(const tt::runtime::Binary &binary, std::string code,
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

  // Checks if the output on the i-th index is a scalar.
  bool isOutputScalar(size_t index) const;

private:
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
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

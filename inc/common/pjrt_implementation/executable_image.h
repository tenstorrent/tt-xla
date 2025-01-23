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

#include "tt/runtime/types.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

class ExecutableImage {

public:
  ExecutableImage(const tt::runtime::Binary &binary, std::string code,
                  size_t arg_count, size_t result_count)
      : ref_count(1), binary(binary), code(code), arg_count(arg_count),
        result_count(result_count) {}
  operator PJRT_Executable *() {
    return reinterpret_cast<PJRT_Executable *>(this);
  }
  static ExecutableImage *Unwrap(PJRT_Executable *exe) {
    return reinterpret_cast<ExecutableImage *>(exe);
  }
  static void BindApi(PJRT_Api *api);

  void AddRef() { ref_count.fetch_add(1); }
  void DecRef() {
    if (ref_count.fetch_sub(1) == 0) {
      delete this;
    }
  }

  const size_t get_arg_count() const { return arg_count; }

  const size_t get_result_count() const { return result_count; }

  const tt::runtime::Binary &get_binary() const { return binary; }

  const std::string &get_code() const { return code; }

private:
  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> ref_count;

  // Raw compiler output.
  tt::runtime::Binary binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string code;

  size_t arg_count;
  size_t result_count;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

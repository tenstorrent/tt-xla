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

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

class ExecutableImage {

public:
  ExecutableImage(std::shared_ptr<void> binary, std::string code,
                  size_t arg_count, size_t result_count,
                  size_t num_addressable_devices)
      : ref_count(1), binary(std::move(binary)), code(code),
        arg_count(arg_count), result_count(result_count),
        num_addressable_devices(num_addressable_devices) {}
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

  std::shared_ptr<void> get_binary() { return binary; }

  const std::string &get_code() const { return code; }

  const size_t get_num_addresible_devices() const {
    return num_addressable_devices;
  }

private:
  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> ref_count;

  // Raw compiler output.
  std::shared_ptr<void> binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string code;

  size_t arg_count;
  size_t result_count;
  size_t num_addressable_devices;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

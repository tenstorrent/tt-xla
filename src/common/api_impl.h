// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_SRC_COMMON_API_IMPL_H_
#define TT_XLA_SRC_COMMON_API_IMPL_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/module_builder.h"
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_description.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"
#include "common/pjrt_implementation/utils.h"
#include "common/platform.h"
#include "tt/runtime/runtime.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

struct ExecutableImage {
  ExecutableImage(std::shared_ptr<void> binary, std::string code,
                  size_t arg_count, size_t result_count)
      : ref_count(1), binary(std::move(binary)), code(code),
        arg_count(arg_count), result_count(result_count) {}
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

private:
  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> ref_count;

public:
  // Raw compiler output.
  std::shared_ptr<void> binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string code;

  size_t arg_count;
  size_t result_count;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

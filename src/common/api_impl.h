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
#include "common/platform.h"
#include "pjrt_implementation/buffer_instance.h"
#include "pjrt_implementation/client_instance.h"
#include "pjrt_implementation/device_description.h"
#include "pjrt_implementation/device_instance.h"
#include "pjrt_implementation/error_instance.h"
#include "pjrt_implementation/event_instance.h"
#include "pjrt_implementation/utils.h"
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

//===----------------------------------------------------------------------===//
// API binding
//===----------------------------------------------------------------------===//

// Binds all monomorphic API members and top-level API struct setup.
void BindMonomorphicApi(PJRT_Api *api);

// Fully binds the PJRT_Api struct for all types. Polymorphic types must be
// specified by template parameters.
template <typename PlatformTy, typename ClientInstanceTy>
static void BindApi(PJRT_Api *api) {
  BindMonomorphicApi(api);

  // Bind polymorphic entry-points.
  api->PJRT_Client_Create = +[](PJRT_Client_Create_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Client_Create");
    auto platform = std::make_unique<PlatformTy>();

    // Populate config_vars() from the client create_options.
    for (size_t i = 0; i < args->num_options; ++i) {
      DLOG_F(WARNING, "Unused config var: %s", args->create_options[i].name);
    }

    auto status = platform->Initialize();
    if (!tt_pjrt_status_is_ok(status)) {
      return MakeError(status);
    }

    auto client = std::make_unique<ClientInstanceTy>(std::move(platform));
    auto *error = client->Initialize();
    if (error)
      return error;

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client *>(client.release());
    return nullptr;
  };
}

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

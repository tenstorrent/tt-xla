// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <utility>

// tt-xla includes
#include "common/pjrt_implementation/error_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_API_BINDINGS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_API_BINDINGS_H_

namespace tt::pjrt {

// Forward declaration
class ClientInstance;

// Global client registry for external access (e.g., from Python via ctypes)
extern ClientInstance* g_last_created_client;

// Top-level API bindings.
void BindMonomorphicApi(PJRT_Api *api);

void BindUndefineds(PJRT_Api *api);

// Initializes and returns PJRT plugin attributes.
PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args);

template <typename PlatformTy, typename ClientInstanceTy>
void BindApi(PJRT_Api *api) {
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
      return *ErrorInstance::makeError(status).release();
    }

    auto client = std::make_unique<ClientInstanceTy>(std::move(platform));
    auto *error = client->Initialize();
    if (error)
      return error;

    // Register client globally for external access
    g_last_created_client = client.get();

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client *>(client.release());
    return nullptr;
  };
}

} // namespace tt::pjrt

#endif

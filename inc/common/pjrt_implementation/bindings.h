// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <utility>

#include "common/pjrt_implementation/error_instance.h"

#ifndef TT_XLA_BINDINGS_H_
#define TT_XLA_BINDINGS_H_

namespace tt::pjrt {

// Top-level API bindings.
void BindMonomorphicApi(PJRT_Api *api);

static void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(LOG_DEBUG, "STUB: " #API);                                          \
    return (decltype(api->API(args)))ErrorInstance::MakeError(                 \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "common/pjrt_implementation/stubs.inc"
}

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
      return ErrorInstance::MakeError(status);
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

#endif

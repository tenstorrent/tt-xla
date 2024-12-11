// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <utility>

#include "common/pjrt_implementation/error_instance.h"
#include "tt/runtime/runtime.h"

#ifndef TT_XLA_UTILS_H_
#define TT_XLA_UTILS_H_
namespace tt::pjrt {

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType);

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type);

// Top-level API bindings.
void BindMonomorphicApi(PJRT_Api *api);

template <typename PlatformTy, typename ClientInstanceTy>
static void BindApi(PJRT_Api *api);

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

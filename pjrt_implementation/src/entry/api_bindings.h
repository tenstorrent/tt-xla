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
#include "api/error_instance.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_API_BINDINGS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_API_BINDINGS_H_

namespace tt::pjrt {

// Top-level API bindings.
void BindMonomorphicApi(PJRT_Api *api);

void BindUndefineds(PJRT_Api *api);

// Initializes and returns PJRT plugin attributes.
PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args);

void BindApi(PJRT_Api *api);

} // namespace tt::pjrt

#endif

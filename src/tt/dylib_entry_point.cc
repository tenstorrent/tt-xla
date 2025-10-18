// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice: SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/dylib_platform.h"
#include "common/pjrt_implementation/api_bindings.h"
#include "common/pjrt_implementation/client_instance.h"
#include "tt/client.h"

// Provides the shared library exports.
#include "common/dylib_entry_point.cc.inc"

namespace tt::pjrt {
namespace {

// Declared but not implemented by the include file.
void InitializeAPI(PJRT_Api *api) {
  BindApi<DylibPlatform, device::TTClientInstance>(api);
}

} // namespace
} // namespace tt::pjrt

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/dylib_platform.h"
#include "tt/client.h"

// Provides the shared library exports.
#include "common/dylib_entry_point.cc.inc"

namespace tt::pjrt {
namespace {

// Declared but not implemented by the include file.
void InitializeAPI(PJRT_Api* api) {
  BindApi<DylibPlatform, device::TTClientInstance>(api);
}

}  // namespace
}  // namespace tt::pjrt

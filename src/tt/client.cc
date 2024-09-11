// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tt/client.h"
#include "common/status.h"
#include <iostream>


namespace tt::pjrt::device {

TTClientInstance::TTClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  DLOG_F(LOG_DEBUG, "TTClientInstance::TTClientInstance");
  cached_platform_name_ = "tt";
}

tt_pjrt_status TTClientInstance::InitializeDeps() {
  DLOG_F(LOG_DEBUG, "TTClientInstance::InitializeDeps");
  return tt_pjrt_status::kSuccess;
}


}  // namespace tt::pjrt::device

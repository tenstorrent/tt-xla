// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

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

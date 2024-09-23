// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_
#define IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_

#include <memory>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "common/status.h"

namespace tt::pjrt {

//===----------------------------------------------------------------------===//
// Platform
// Encapsulates aspects of the platform which may differ under different
// usage and/or environments.
//===----------------------------------------------------------------------===//

class Platform {
 public:
  virtual ~Platform();
  tt_pjrt_status Initialize();
 protected:
  virtual tt_pjrt_status SubclassInitialize() = 0;
  void InitializeLogging();
};

}  // namespace tt::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_

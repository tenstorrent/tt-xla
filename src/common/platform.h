// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

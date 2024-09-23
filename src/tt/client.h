// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_

#include "common/api_impl.h"

namespace tt::pjrt::device {

class TTClientInstance final : public ClientInstance {
 public:
  TTClientInstance(std::unique_ptr<Platform> platform);
  ~TTClientInstance() {};

 private:
  tt_pjrt_status InitializeDeps();

  // Instance scoped options.
  bool single_threaded_debug_ = false;
};

}  // namespace tt::pjrt::device

#endif  // IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_

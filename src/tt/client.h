// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_

#include "common/api_impl.h"

namespace iree::pjrt::cpu {

class CPUClientInstance final : public ClientInstance {
 public:
  CPUClientInstance(std::unique_ptr<Platform> platform);
  ~CPUClientInstance() {};
  bool SetDefaultCompilerFlags(CompilerJob* compiler_job) override;

 private:
  tt_pjrt_status InitializeDeps();

  // Instance scoped options.
  bool single_threaded_debug_ = false;
};

}  // namespace iree::pjrt::cpu

#endif  // IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_

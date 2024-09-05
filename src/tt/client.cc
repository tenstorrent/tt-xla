// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tt/client.h"
#include "common/status.h"
#include <iostream>


namespace iree::pjrt::cpu {

CPUClientInstance::CPUClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  std::cout << "CPUClientInstance::CPUClientInstance" << std::endl;
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "tt";
}

tt_pjrt_status CPUClientInstance::InitializeDeps() {
  std::cout << "CPUClientInstance::InitializeDeps" << std::endl;
  return tt_pjrt_status::kSuccess;
}

bool CPUClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  return compiler_job->SetFlag("");
}

}  // namespace iree::pjrt::cpu

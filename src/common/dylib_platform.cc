// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/dylib_platform.h"

#include <array>
#include <optional>
#include <string>
#include <iostream>

namespace tt::pjrt {

namespace {

bool InitializeCompilerForProcess(const std::string& library_path) {
  DLOG_F(LOG_DEBUG, "Loading compiler library: %s", library_path.c_str());
  return true;
}

// Since we delay load the compiler, it can only be done once per process.
// First one to do it wins. Returns the path of the loaded compiler or
// empty if it could not be loaded.
std::optional<std::string> LoadCompilerStubOnce(
    const std::string& library_path) {
  return {};
}

bool InitializePartitionerForProcess(const std::string& library_path) {
  return true;
}

std::optional<std::string> LoadPartitionerStubOnce(
    const std::string& library_path) {
  return {};
}

}  // namespace

tt_pjrt_status DylibPlatform::SubclassInitialize() {
  DLOG_F(LOG_DEBUG, "DylibPlatform::SubclassInitialize");
  return tt_pjrt_status::kSuccess;
}

std::optional<std::string> DylibPlatform::GetHomeDir() {
  return {};
}

std::optional<std::string> DylibPlatform::GetBinaryDir() {
  return {};
}

std::optional<std::string> DylibPlatform::GetLibraryDir() {
  return {};
}

std::optional<std::string> DylibPlatform::GetCompilerLibraryPath() {
  return {};
}

std::optional<std::string> DylibPlatform::GetPartitionerLibraryPath() {
  return {};
}

}  // namespace tt::pjrt

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/dylib_platform.h"

#include <array>
#include <optional>
#include <string>
#include <iostream>

namespace iree::pjrt {

namespace {

bool InitializeCompilerForProcess(const std::string& library_path) {
  std::cout << "Loading compiler library: " << library_path << std::endl;
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
  std::cout << "DylibPlatform::SubclassInitialize" << std::endl;
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

}  // namespace iree::pjrt

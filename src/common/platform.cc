// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/platform.h"
#include "loguru/loguru.hpp"

#include <cstdlib>
#include <iostream>

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// ConfigVars
//===----------------------------------------------------------------------===//

void ConfigVars::EnableEnvFallback(std::string env_fallback_prefix) {
  env_fallback_prefix_ = env_fallback_prefix;
}

std::optional<std::string> ConfigVars::Lookup(const std::string& key) {
  auto found_it = kv_entries_.find(key);
  if (found_it != kv_entries_.end()) {
    return found_it->second;
  }

  // Env fallback?
  if (!env_fallback_prefix_) return {};

  std::string full_env_key = *env_fallback_prefix_;
  full_env_key.append(key);
  char* found_env = std::getenv(full_env_key.c_str());
  if (found_env) {
    return std::string(found_env);
  }

  return {};
}

void ConfigVars::Set(const std::string& key, std::string value) {
  kv_entries_[key] = std::move(value);
}

//===----------------------------------------------------------------------===//
// Platform
//===----------------------------------------------------------------------===//

Platform::~Platform() = default;

tt_pjrt_status Platform::Initialize() {
  DLOG_F(INFO, "Platform::Initialize");
  
  SubclassInitialize();
  // Default logger.
  logger_ = std::make_unique<Logger>();

  return tt_pjrt_status::kSuccess;
}

}  // namespace iree::pjrt

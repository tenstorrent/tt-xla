// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/platform.h"

#include <cstdlib>
#include <iostream>
#include <cstring>
#include "loguru/loguru.hpp"

namespace tt::pjrt {


//===----------------------------------------------------------------------===//
// Platform
//===----------------------------------------------------------------------===//

Platform::~Platform() = default;

void Platform::InitializeLogging() {
  
  loguru::g_stderr_verbosity = 0;
  const char *loguru_verbosity = std::getenv("LOGGER_LEVEL");
  if (loguru_verbosity) {
    if (strcmp(loguru_verbosity, "DEBUG") == 0) {
      loguru::g_stderr_verbosity = LOG_DEBUG;
    } else if (strcmp(loguru_verbosity, "INFO") == 0) {
      loguru::g_stderr_verbosity = 0;
    } else if (strcmp(loguru_verbosity, "WARNING") == 0) {
      loguru::g_stderr_verbosity = -1;
    } else if (strcmp(loguru_verbosity, "ERROR") == 0) {
      loguru::g_stderr_verbosity = -2;
    }
    else {
      LOG_F(ERROR, "Invalid LOGGER_LEVEL: %s", loguru_verbosity);
    }
  }
}

tt_pjrt_status Platform::Initialize() {
  DLOG_F(LOG_DEBUG, "Platform::Initialize");
  InitializeLogging();
  
  SubclassInitialize();

  return tt_pjrt_status::kSuccess;
}

}  // namespace tt::pjrt

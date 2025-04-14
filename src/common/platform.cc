// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/platform.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "loguru/loguru.hpp"

namespace tt::pjrt {

//===----------------------------------------------------------------------===//
// Platform
//===----------------------------------------------------------------------===//

Platform::~Platform() = default;

void Platform::InitializeLogging() {
  // Everything with a verbosity equal or below g_stderr_verbosity will
  // be written to stderr. Setting to INFO by default.
  loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_INFO;

  const char *loguru_verbosity = std::getenv("LOGGER_LEVEL");
  if (!loguru_verbosity) {
    return;
  }

  if (strcmp(loguru_verbosity, "DEBUG") == 0) {
    loguru::g_stderr_verbosity = LOG_DEBUG;
  } else if (strcmp(loguru_verbosity, "INFO") == 0) {
    loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_INFO;
  } else if (strcmp(loguru_verbosity, "WARNING") == 0) {
    loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_WARNING;
  } else if (strcmp(loguru_verbosity, "ERROR") == 0) {
    loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_ERROR;
  } else {
    LOG_F(ERROR, "Invalid LOGGER_LEVEL: %s", loguru_verbosity);
  }
}

tt_pjrt_status Platform::Initialize() {
  DLOG_F(LOG_DEBUG, "Platform::Initialize");

  InitializeLogging();
  SubclassInitialize();

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt

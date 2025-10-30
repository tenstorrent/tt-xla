// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include <cstdlib>
#include <cstring>

#include "utils/logging.h"

namespace tt::pjrt {

void initializeLogging() {
  // Everything with a verbosity equal or below g_stderr_verbosity will
  // be written to stderr. Setting to INFO by default.
  loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_INFO;

  const char *loguru_verbosity = std::getenv("LOGGER_LEVEL");
  if (!loguru_verbosity) {
    return;
  }

  if (strcmp(loguru_verbosity, "VERBOSE") == 0) {
    loguru::g_stderr_verbosity = LOG_VERBOSE;
  } else if (strcmp(loguru_verbosity, "DEBUG") == 0) {
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

  // Check for log file configuration - only if LOGGER_LEVEL is set
  const char *log_file_path = std::getenv("TT_XLA_LOGGER_FILE");
  if (log_file_path) {
    // Add file output using the same verbosity level as stderr
    loguru::add_file(log_file_path, loguru::Append, loguru::g_stderr_verbosity);
  }
}

} // namespace tt::pjrt

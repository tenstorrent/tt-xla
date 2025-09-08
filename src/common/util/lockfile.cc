// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/util/lockfile.h"

// c++ standard library includes
#include <cstdio>
#include <stdexcept>

// loguru includes
#include "loguru/loguru.hpp"

namespace tt::util {

LockFile::LockFile(const std::filesystem::path &lock_path)
    : m_lock_path(lock_path), m_lock_file(nullptr) {

  // Ensure the parent directory exists
  std::filesystem::create_directories(m_lock_path.parent_path());

  // Try to create the lock file using C file API with "wx" mode
  // "wx" mode creates file for writing and fails if it already exists
  m_lock_file = std::fopen(m_lock_path.c_str(), "wx");
  if (m_lock_file == nullptr) {
    ABORT_F("Failed to acquire lock: %s (file may already exist or permission "
            "denied)",
            m_lock_path.c_str());
  }
}

LockFile::~LockFile() {
  std::fclose(m_lock_file);
  std::filesystem::remove(m_lock_path);
}

} // namespace tt::util

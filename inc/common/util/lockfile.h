// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_FILE_UTILS_H_
#define TT_XLA_INC_COMMON_FILE_UTILS_H_

// c++ standard library includes
#include <cstdio>
#include <filesystem>

namespace tt::util {

// RAII lock between processes using lock files.
class LockFile {
public:
  explicit LockFile(const std::filesystem::path &lock_path);

  ~LockFile();

  LockFile(const LockFile &) = delete;
  LockFile &operator=(const LockFile &) = delete;
  LockFile(LockFile &&) = delete;
  LockFile &operator=(LockFile &&) = delete;

private:
  std::filesystem::path m_lock_path;
  FILE *m_lock_file; // Uses C API because std::ios::noreplace is C++23 only.
};

} // namespace tt::util

#endif // TT_XLA_INC_COMMON_FILE_UTILS_H_

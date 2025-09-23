// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

#include <sstream>
#include <vector>

namespace tt::pjrt::utils {

template <typename T> std::string to_string(const std::vector<T> vec) {
  std::stringstream res;
  res << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res << vec[i] << (i + 1 < vec.size() ? ", " : "");
  }
  res << "]";
  return res.str();
}

} // namespace tt::pjrt::utils

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_

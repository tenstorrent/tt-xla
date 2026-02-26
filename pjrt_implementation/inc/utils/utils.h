// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_UTILS_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_UTILS_H_

#include "logging.h"
#include <functional>
#include <optional>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>

namespace tt::pjrt::utils {

// Converts a vector to a string representation for printing its content.
template <typename T> std::string to_string(const std::vector<T> vec) {
  std::stringstream res;
  res << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res << vec[i] << (i + 1 < vec.size() ? ", " : "");
  }
  res << "]";
  return res.str();
}

// Invokes function with arguments inside try/catch block.
// Return type is std::optional with derived type from calling function, so user
// can check return value for nullopt (or with implicit bool operator).
// If function return type is void, std::optional will hold std::monostate.
//
// Example usage:
// std::vector<int> my_fn(int first, int second);
//
// auto r = utils::invoke_noexcept(my_fn, arg1, arg2);
// if (!r)
//  ... failed - handle error ....
//
// or keep going if successful:
// std::vector<int>& output = *r;
// ...
//
// Also, you can invoke it with lambda:
// auto r = utils::invoke_noexcept([&] { return my_fn(arg1, arg2); });
//
template <class Fn, class... Args,
          class ReturnType = std::invoke_result_t<Fn, Args...>>
std::optional<std::conditional_t<std::is_same_v<ReturnType, void>,
                                 std::monostate, ReturnType>>
invoke_noexcept(Fn &&fn, Args &&...args) noexcept {
  DLOG_F(LOG_DEBUG, "Asif:: utils_0 ");
  try {
    DLOG_F(LOG_DEBUG, "Asif:: utils_1 ");
    if constexpr (std::is_same_v<ReturnType, void>) {
      DLOG_F(LOG_DEBUG, "Asif:: utils_2 ");
      std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
      DLOG_F(LOG_DEBUG, "Asif:: utils_3 ");
      return std::optional{std::monostate{}};
    } else {
      DLOG_F(LOG_DEBUG, "Asif:: utils_4 ");
      auto temp = std::optional{
          std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...)};
      DLOG_F(LOG_DEBUG, "Asif:: utils_4_a ");
      return temp;
    }
    DLOG_F(LOG_DEBUG, "Asif:: utils_5_x ");
  } catch (const std::exception &ex) {
    DLOG_F(LOG_DEBUG, "Asif:: utils_7 ");
    LOG_F(ERROR, "Exception:\n{%s}\n", ex.what());
  } catch (...) {
    DLOG_F(LOG_DEBUG, "Asif:: utils_8 ");
    LOG_F(ERROR, "Unknown exception.");
  }
  DLOG_F(LOG_DEBUG, "Asif:: utils_6_x ");

  return std::nullopt;
}

} // namespace tt::pjrt::utils

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_UTILS_H_

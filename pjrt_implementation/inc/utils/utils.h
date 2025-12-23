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
  try {
    if constexpr (std::is_same_v<ReturnType, void>) {
      std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
      return std::optional{std::monostate{}};
    } else {
      return std::optional{
          std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...)};
    }
  } catch (const std::exception &ex) {
    DLOG_F(ERROR, "Exception:\n{%s}\n", ex.what());
  } catch (...) {
    DLOG_F(ERROR, "Unknown exception.");
  }

  return std::nullopt;
}

} // namespace tt::pjrt::utils

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_UTILS_H_

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//  Assert header summary:
//
//  TT_ASSERT -> debug build only assert. Throws exception with condition,
//  location and backtrace.
//
//  TT_FATAL -> always assert. Throws exception with condition, location and
//  backtrace.
//
//  TT_THROW -> always throws exception with condition, location and backtrace.
//
//  Environment variables:
//  TT_XLA_DISABLE_BACKTRACE -> disables backtrace.
//  TT_XLA_ASSERT_ABORT -> forces abort instead of throw.
//
//
//  This header is taken from tt-metal repo and slightly modified (we don't have
//  fmt, nor c++20). String formatting is done via loguru.
#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_ASSERT_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_ASSERT_H_

// c++ standard library includes
#include <cxxabi.h>
#include <execinfo.h>
#include <format>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// tt-xla includes
#include "logging.h"

namespace tt::assert {

namespace detail {
static std::string demangle(const char *str);
}

// @brief Get the current call stack
// @param[out] bt Save Call Stack
// @param[in] size Maximum number of return layers
// @param[in] skip Skip the number of layers at the top of the stack
// NOLINTBEGIN(cppcoreguidelines-no-malloc)
inline std::vector<std::string> backtrace(int size = 64, int skip = 1) {

  std::vector<std::string> bt;
  bt.reserve(size - skip);

  void **array = (void **)malloc((sizeof(void *) * size));
  size_t bt_size = ::backtrace(array, size);
  char **strings = backtrace_symbols(array, bt_size);
  if (strings == nullptr) {
    LOG_F(ERROR, "backtrace_symbols error.");
    free(array); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    return bt;
  }

  for (size_t i = skip; i < bt_size; ++i)
    bt.push_back(detail::demangle(strings[i]));

  free(strings); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
  free(array);   // NOLINT(bugprone-multi-level-implicit-pointer-conversion)

  return bt;
}
// NOLINTEND(cppcoreguidelines-no-malloc)

// @brief String to get current stack information
// @param[in] size Maximum number of stacks
// @param[in] skip Skip the number of layers at the top of the stack
// @param[in] prefix Output before stack information
inline std::string backtrace_to_string(int size = 64, int skip = 2,
                                       const std::string &prefix = "") {

  std::vector<std::string> bt{backtrace(size, skip)};
  std::stringstream ss;
  for (const auto &line : bt)
    ss << prefix << line << '\n';

  return ss.str();
}

namespace detail {
// NOLINTBEGIN(cppcoreguidelines-no-malloc)
static std::string demangle(const char *str) {

  size_t size = 0;
  int status = 0;
  std::string rt(256, '\0');
  if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", rt.data())) {
    char *v = abi::__cxa_demangle(rt.data(), nullptr, &size, &status);
    if (v) {
      std::string result{v};
      free(v);
      return result;
    }
  }
  return str;
}
// NOLINTEND(cppcoreguidelines-no-malloc)

// Since we don't have compile time format checking, we must use printf-like
// formatting. Clang can not check format and args through function call even
// when they are perfectly forwarded, so we must disable format-security
// warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"

template <typename... Args>
[[noreturn]] void tt_throw(const char *file, int line, const char *assert_type,
                           const char *cond_str, Args &&...args) {

  if (std::getenv("TT_XLA_ASSERT_ABORT")) {
    LOG_F(ERROR, "%s: %s", assert_type, cond_str);
    if constexpr (sizeof...(args) > 0)
      LOG_F(ERROR, std::forward<Args>(args)...);

    abort();
  }

  std::stringstream trace;
  trace << assert_type << " @ " << file << ":" << line << ": " << cond_str
        << '\n';

  if constexpr (sizeof...(args) > 0)
    trace << "info:\n"
          << loguru::textprintf(std::forward<Args>(args)...).c_str() << '\n';

  LOG_F(ERROR, trace.str().c_str());

  if (!std::getenv("TT_XLA_DISABLE_BACKTRACE")) {
    trace << "backtrace:\n";
    trace << tt::assert::backtrace_to_string(100, 3, " --- ");
  }

  throw std::runtime_error(trace.str());
}

#pragma clang diagnostic pop

template <typename... Args>
void tt_assert(char const *file, int line, char const *assert_type, bool cond,
               char const *cond_str, Args &&...args) {

  if (!cond) [[unlikely]]
    tt_throw(file, line, assert_type, cond_str, std::forward<Args>(args)...);
}

} // namespace detail
} // namespace tt::assert

#ifdef DEBUG
#ifndef TT_ASSERT
#define TT_ASSERT(condition, ...)                                              \
  do {                                                                         \
    if (!(condition)) [[unlikely]]                                             \
      tt::assert::detail::tt_assert(__FILE__, __LINE__, "TT_ASSERT",           \
                                    (condition), #condition, ##__VA_ARGS__);   \
  } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif // TT_ASSERT
#else
#define TT_ASSERT(condition, ...)                                              \
  do {                                                                         \
    (void)(condition);                                                         \
  } while (0) // this was done to avoid the compiler flagging unused variables
              // when building Release
#endif // DEBUG

#ifndef TT_FATAL
#define TT_FATAL(condition, ...)                                               \
  do {                                                                         \
    if (!(condition)) [[unlikely]] {                                           \
      tt::assert::detail::tt_throw(__FILE__, __LINE__, "TT_FATAL", #condition, \
                                   ##__VA_ARGS__);                             \
      __builtin_unreachable();                                                 \
    }                                                                          \
  } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif // TT_FATAL

#ifndef TT_THROW
#define TT_THROW(...)                                                          \
  tt::assert::detail::tt_throw(__FILE__, __LINE__, "TT_THROW",                 \
                               "tt::exception", ##__VA_ARGS__)
#endif // TT_THROW

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_UTILS_ASSERT_H_

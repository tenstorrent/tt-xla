// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#include "tt/runtime/debug.h"
#endif

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
namespace py = pybind11;

PYBIND11_MODULE(tt_xla_debug, m) {
  #if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  py::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get_debug_hooks",
          [](py::function func) {
            return tt::runtime::debug::Hooks::get(
                std::nullopt,
                [func](tt::runtime::Binary binary,
                       tt::runtime::CallbackContext programContext,
                       tt::runtime::OpContext opContext) {
                  func(binary, programContext, opContext);
                });
          },
          "Get the debug hooks")
      .def("__str__", [](const tt::runtime::debug::Hooks &hooks) {
        std::stringstream os;
        os << hooks;
        return os.str();
      });

  /**
   * Cleanup code to force a well ordered destruction w.r.t. the GIL
   */
   auto cleanup_callback = []() {
    tt::runtime::debug::Hooks::get().unregisterHooks();
  };
  m.add_object("_cleanup", py::capsule(cleanup_callback));
  m.def("unregister_hooks",
        []() { tt::runtime::debug::Hooks::get().unregisterHooks(); });
  m.def("is_runtime_debug_enabled", []() -> bool { return true; });
#else
  m.def("is_runtime_debug_enabled", []() -> bool { return false; });
#endif
}
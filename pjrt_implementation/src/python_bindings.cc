// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "ttmlir/Target/Common/types_generated.h" // Needed for EnumNameArch
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(native, m) {
  m.doc() = "TT-XLA Native Python Bindings";

  m.def(
      "get_arch",
      []() { return tt::target::EnumNameArch(tt::runtime::getArch()); },
      "Get the architecture of the device as a string");
}

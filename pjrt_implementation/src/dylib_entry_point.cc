// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice: SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/event_instance.h"
#include "api_bindings.h"

// Provides the shared library exports.
#include "dylib_entry_point.cc.inc"

namespace tt::pjrt {
namespace {

// Declared but not implemented by the include file.
void InitializeAPI(PJRT_Api *api) { bindApi(api); }

} // namespace
} // namespace tt::pjrt

// Performs controlled shutdown of plugin-owned resources that must be torn
// down while the host Python interpreter (GIL + modules) is still alive.
// Invoked from Python via `atexit` / ctypes before interpreter finalization
// tears down state that in-flight callbacks depend on.
extern "C" PJRT_PLUGIN_EXPORTED void tt_pjrt_shutdown() {
  tt::pjrt::EventInstance::destroyCallbackWorker();
}

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "xla/pjrt/c/pjrt_c_api.h"

#include <memory>
#include <vector>

#include "common/module_builder.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/memory_instance.h"
#include "common/pjrt_implementation/loaded_executable_instance.h"
#include "common/platform.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_

namespace tt::pjrt {

//===----------------------------------------------------------------------===//
// ClientInstance
// The root of the runtime hierarchy, these map to an IREE driver and are
// created against an API.
//===----------------------------------------------------------------------===//
class ClientInstance {

public:
  ClientInstance(std::unique_ptr<Platform> platform);
  virtual ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void BindApi(PJRT_Api *api);

  static ClientInstance *Unwrap(PJRT_Client *client) {
    return reinterpret_cast<ClientInstance *>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error *Initialize();

  Platform &platform() { return *platform_; }
  const std::vector<DeviceInstance *> &devices() { return devices_; }
  const std::vector<DeviceInstance *> &addressable_devices() {
    return addressable_devices_;
  }
  const std::string &cached_platform_name() { return cached_platform_name_; }
  const std::string &cached_platform_version() {
    return cached_platform_version_;
  }

  // Compiles.
  // See TODOs in PJRT_Client_Compile.
  PJRT_Error *
  Compile(const PJRT_Program *program, /*xla::CompileOptions options, */
          LoadedExecutableInstance **executable);

  // Advances the timeline, returning (current, next) time point values.
  std::tuple<uint64_t, uint64_t> AdvanceTimeline();

protected:
  std::string cached_platform_name_;
  std::string cached_platform_version_;

private:
  tt_pjrt_status InitializeCompiler();
  tt_pjrt_status PopulateDevices();
  tt_pjrt_status PopulateMemories();

  std::unique_ptr<Platform> platform_;

  std::vector<DeviceInstance *> devices_;
  std::vector<DeviceInstance *> addressable_devices_;
  std::vector<MemoryInstance *> m_memories;

  std::unique_ptr<ModuleBuilder> module_builder_;

  // System descriptor (that TTIR to TTNN backend pipeline needs).
  tt::runtime::SystemDesc system_descriptor_;

  // TODO: Remove once tt-mlir supports passing the system descriptor object to
  // TTIR to TTNN backend pipeline.
  std::string cached_system_descriptor_path_;

  // Synchronization.
  // We keep one global execution timeline across all devices. The management
  // of this is currently somewhat primitive: we increment it by one for each
  // invocation. Batch invocations (i.e. across multiple devices), only
  // increment by one. In the future, additional parallelism could be plumbed
  // up to the framework to allow different kinds of timeline management.
  // Waiting on the current value of |execution_timeline_| will drain all
  // scheduled work to date.
  uint64_t execution_timeline_ = 0ull;
};

} // namespace tt::pjrt

#endif

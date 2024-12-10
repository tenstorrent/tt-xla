// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_SRC_COMMON_API_IMPL_H_
#define TT_XLA_SRC_COMMON_API_IMPL_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/module_builder.h"
#include "common/platform.h"
#include "pjrt_implementation/buffer_instance.h"
#include "pjrt_implementation/device_description.h"
#include "pjrt_implementation/device_instance.h"
#include "pjrt_implementation/error_instance.h"
#include "pjrt_implementation/utils.h"
#include "tt/runtime/runtime.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt {

class ClientInstance;
class EventInstance;
//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

class EventInstance {
public:
  EventInstance();
  ~EventInstance();
  operator PJRT_Event *() { return reinterpret_cast<PJRT_Event *>(this); }
  static void BindApi(PJRT_Api *api);
  static EventInstance *Unwrap(PJRT_Event *exe) {
    return reinterpret_cast<EventInstance *>(exe);
  }

  tt_pjrt_status OnReady(PJRT_Event_OnReadyCallback callback, void *user_arg);
  ErrorInstance *error();
  bool is_ready();

private:
  void SignalReady(tt_pjrt_status status);

  std::mutex lock_;
  tt_pjrt_status status_ = tt_pjrt_status::kSuccess;
  bool is_ready_;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void *>> pending_callbacks_;
  std::unique_ptr<std::thread> signal_thread_;
};

struct ExecutableImage {
  ExecutableImage(std::shared_ptr<void> binary, std::string code,
                  size_t arg_count, size_t result_count)
      : ref_count(1), binary(std::move(binary)), code(code),
        arg_count(arg_count), result_count(result_count) {}
  operator PJRT_Executable *() {
    return reinterpret_cast<PJRT_Executable *>(this);
  }
  static ExecutableImage *Unwrap(PJRT_Executable *exe) {
    return reinterpret_cast<ExecutableImage *>(exe);
  }
  static void BindApi(PJRT_Api *api);

  void AddRef() { ref_count.fetch_add(1); }
  void DecRef() {
    if (ref_count.fetch_sub(1) == 0) {
      delete this;
    }
  }

private:
  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> ref_count;

public:
  // Raw compiler output.
  std::shared_ptr<void> binary;

  // Original code fed to the compiler. Stored for debugging.
  const std::string code;

  size_t arg_count;
  size_t result_count;
};

struct ResidentExecutable {
  DeviceInstance *device_instance;
  size_t arg_count;
  size_t result_count;
};

class LoadedExecutableInstance {
public:
  LoadedExecutableInstance(
      ClientInstance &client, ExecutableImage *image,
      const std::vector<DeviceInstance *> &addressable_devices)
      : client_(client), image_(image),
        addressable_devices_(addressable_devices) {}
  ~LoadedExecutableInstance() { image_->DecRef(); }

  operator PJRT_LoadedExecutable *() {
    return reinterpret_cast<PJRT_LoadedExecutable *>(this);
  }
  static void BindApi(PJRT_Api *api);
  static LoadedExecutableInstance *Unwrap(PJRT_LoadedExecutable *exe) {
    return reinterpret_cast<LoadedExecutableInstance *>(exe);
  }

  const std::vector<DeviceInstance *> &addressable_devices() {
    return addressable_devices_;
  }

  // Loads all executables to addressable devices.
  tt_pjrt_status LoadAll();

  tt_pjrt_status GetDefaultResidentExecutable(ResidentExecutable **out_loaded);
  tt_pjrt_status GetArgResultCount(size_t *out_arg_count,
                                   size_t *out_result_count);

  tt_pjrt_status Execute(PJRT_LoadedExecutable_Execute_Args *args);

private:
  ClientInstance &client_;
  ExecutableImage *image_; // Ref-counted semantics.
  std::vector<DeviceInstance *> addressable_devices_;
  std::vector<ResidentExecutable> resident_executables_;
};

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

  std::unique_ptr<Platform> platform_;

  std::vector<DeviceInstance *> devices_;
  std::vector<DeviceInstance *> addressable_devices_;

  std::unique_ptr<ModuleBuilder> module_builder_;

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

//===----------------------------------------------------------------------===//
// API binding
//===----------------------------------------------------------------------===//

// Binds all monomorphic API members and top-level API struct setup.
void BindMonomorphicApi(PJRT_Api *api);

// Fully binds the PJRT_Api struct for all types. Polymorphic types must be
// specified by template parameters.
template <typename PlatformTy, typename ClientInstanceTy>
static void BindApi(PJRT_Api *api) {
  BindMonomorphicApi(api);

  // Bind polymorphic entry-points.
  api->PJRT_Client_Create = +[](PJRT_Client_Create_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Client_Create");
    auto platform = std::make_unique<PlatformTy>();

    // Populate config_vars() from the client create_options.
    for (size_t i = 0; i < args->num_options; ++i) {
      DLOG_F(WARNING, "Unused config var: %s", args->create_options[i].name);
    }

    auto status = platform->Initialize();
    if (!tt_pjrt_status_is_ok(status)) {
      return MakeError(status);
    }

    auto client = std::make_unique<ClientInstanceTy>(std::move(platform));
    auto *error = client->Initialize();
    if (error)
      return error;

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client *>(client.release());
    return nullptr;
  };
}

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_API_IMPL_H_

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

// c++ standard library includes
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

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/module_builder.h"
#include "common/platform.h"

namespace tt::pjrt {

class ClientInstance;
class DeviceInstance;
class ErrorInstance;
class EventInstance;

class ErrorInstance {
public:
  ErrorInstance(tt_pjrt_status status) : status_(status) {}
  ~ErrorInstance() {}
  static void BindApi(PJRT_Api *api);

  static const ErrorInstance *FromError(const PJRT_Error *error) {
    return reinterpret_cast<const ErrorInstance *>(error);
  }

  tt_pjrt_status status() const { return status_; }
  const std::string &message() const;

private:
  tt_pjrt_status status_;
  mutable std::string cached_message_;
};

inline PJRT_Error *MakeError(tt_pjrt_status status) {
  if (tt_pjrt_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error *>(alloced_error.release());
}

//===----------------------------------------------------------------------===//
// BufferInstance
//===----------------------------------------------------------------------===//
class BufferInstance {
public:
  BufferInstance(DeviceInstance &device, tt::runtime::Tensor tensor,
                 std::vector<std::uint32_t> shape,
                 std::vector<std::uint32_t> stride);
  BufferInstance(DeviceInstance &device);
  ~BufferInstance();
  operator PJRT_Buffer *() { return reinterpret_cast<PJRT_Buffer *>(this); }
  static BufferInstance *Unwrap(PJRT_Buffer *buffer) {
    return reinterpret_cast<BufferInstance *>(buffer);
  }
  static void BindApi(PJRT_Api *api);

  // iree_hal_buffer_view_t* buffer_view() { return buffer_view_.get(); }
  DeviceInstance &device() { return device_; }
  tt_pjrt_status AsyncDeallocate();
  tt_pjrt_status Delete();
  bool is_deleted() { return is_deleted_; }
  bool is_on_cpu() {
    // TODO: Plumb through an indication if running on CPU and then implement
    // the hook to get an unsafe pointer (avoids a copy).
    return false;
  }
  tt::runtime::Tensor tensor() { return tensor_.value(); }

  PJRT_Error *GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args *args);
  // Gets the required host size in bytes to copy to host.
  tt_pjrt_status GetHostSizeInBytes(size_t *host_size);
  tt_pjrt_status CopyToHost(void *dst, size_t dst_size,
                            EventInstance **done_event);

  const int64_t *dims() { return dims_.data(); }
  size_t num_dims() { return dims_.size(); }
  void setType(PJRT_Buffer_Type Type) { DataType = Type; }
  std::optional<PJRT_Buffer_Type> getType() { return DataType; }

  // Get the data type for a tensor through runtime if DataType is not set.
  PJRT_Buffer_Type getRuntimeType();

  int unique_id() { return unique_id_; }

private:
  static int id_counter_;
  int unique_id_;
  void ComputeLayout();

  DeviceInstance &device_;
  // When the buffer resource gets freed, this is set to true.
  bool is_deleted_ = false;

  // API elements that must have the same lifetime as BufferInstance.
  std::vector<int64_t> dims_;
  std::vector<std::uint32_t> stride_;
  std::optional<tt::runtime::Tensor> tensor_;

  std::vector<int64_t> minor_to_major_;
  std::vector<int64_t> tile_dims_;
  std::vector<size_t> tile_dim_sizes_;

  // Underlying datatype of tensor.
  std::optional<PJRT_Buffer_Type> DataType;
};

//===----------------------------------------------------------------------===//
// DeviceDescription
//===----------------------------------------------------------------------===//

class DeviceDescription {
public:
  DeviceDescription(int32_t client_id) : client_id_(client_id) {};
  ~DeviceDescription();
  operator PJRT_DeviceDescription *() {
    return reinterpret_cast<PJRT_DeviceDescription *>(this);
  }
  static void BindApi(PJRT_Api *api);

  static DeviceDescription *Unwrap(PJRT_DeviceDescription *device) {
    return reinterpret_cast<DeviceDescription *>(device);
  }

  std::string_view kind_string() { return kind_string_; }
  std::string_view debug_string() { return debug_string_; }
  std::string_view user_string() {
    std::stringstream ss;
    ss << "TTDevice(id=" << device_id() << ")";
    user_string_ = ss.str();
    return user_string_;
  }
  // TODO
  int64_t device_id() { return 0; }

  int client_id() { return client_id_; }

  int process_index() { return 0; }

private:
  int client_id_;
  std::string kind_string_ = "wormhole";
  std::string debug_string_ = "debug_string";
  std::string user_string_ = "";
};

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

class DeviceInstance {
public:
  DeviceInstance(int client_id, ClientInstance &client)
      : client_(client), description_(client_id) {}
  ~DeviceInstance();
  operator PJRT_Device *() { return reinterpret_cast<PJRT_Device *>(this); }
  static void BindApi(PJRT_Api *api);

  static DeviceInstance *Unwrap(PJRT_Device *device) {
    return reinterpret_cast<DeviceInstance *>(device);
  }

  static DeviceInstance *Unwrap(PJRT_DeviceDescription *device_description) {
    return reinterpret_cast<DeviceInstance *>(device_description);
  }
  ClientInstance &client() { return client_; }
  bool is_addressable() { return true; }
  int local_hardware_id() { return -1; }

  tt_pjrt_status
  HostBufferToDeviceZeroDim(PJRT_Buffer_Type type, const int64_t *dims,
                            size_t num_dims,
                            EventInstance **out_done_with_host_buffer_event,
                            BufferInstance **out_buffer);

  tt_pjrt_status
  HostBufferToDeviceSplat(const void *data, PJRT_Buffer_Type type,
                          const int64_t *dims, size_t num_dims,
                          EventInstance **out_done_with_host_buffer_event,
                          BufferInstance **out_buffer);

  tt_pjrt_status
  HostBufferToDevice(const void *data, PJRT_Buffer_Type type,
                     const int64_t *dims, size_t num_dims,
                     const int64_t *byte_strides, size_t num_byte_strides,
                     PJRT_HostBufferSemantics host_buffer_semantics,
                     EventInstance **out_done_with_host_buffer_event,
                     BufferInstance **out_buffer);

  DeviceDescription *device_description() { return &description_; }

private:
  tt_pjrt_status OpenDevice();

  ClientInstance &client_;
  uint64_t last_transfer_timepoint_ = 0;
  DeviceDescription description_;
};

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

  // Returns the module builder used for this ClientInstance.
  const ModuleBuilder *get_module_builder() const {
    return module_builder_.get();
  }

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

// Initializes and returns PJRT plugin attributes.
PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args);

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

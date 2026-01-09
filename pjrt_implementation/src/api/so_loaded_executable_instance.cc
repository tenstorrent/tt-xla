// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/so_loaded_executable_instance.h"

// c++ standard library includes
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>

// POSIX includes
#include <dlfcn.h>

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "utils/data_type_utils.h"
#include "utils/logging.h"
#include "utils/status.h"

namespace tt::pjrt {

// PythonRunnerHandler implementation

PythonRunnerHandler::PythonRunnerHandler()
    : m_initialized(false), m_handle(nullptr), m_create_runner(nullptr),
      m_destroy_runner(nullptr), m_add_to_sys_path(nullptr),
      m_load_module(nullptr), m_forward(nullptr) {}

PythonRunnerHandler::~PythonRunnerHandler() {
  if (m_handle != nullptr) {
    dlclose(m_handle);
  }
}

std::optional<std::string> PythonRunnerHandler::findPythonRunnerLibraryPath() {
  const char *mlir_home = std::getenv("TT_MLIR_HOME");
  if (mlir_home == nullptr) {
    return std::nullopt;
  }

  std::string runner_lib_path =
      std::string(mlir_home) + "/build/lib/libtt-alchemist-python-runner.so";

  if (std::filesystem::exists(runner_lib_path)) {
    return runner_lib_path;
  }

  return std::nullopt;
}

void PythonRunnerHandler::initialize() {
  std::optional<std::string> maybe_so_path = findPythonRunnerLibraryPath();
  if (!maybe_so_path.has_value()) {
    DLOG_F(WARNING,
           "tt-alchemist-python-runner library not found in TT_MLIR_HOME");
    return;
  }
  std::string so_path = maybe_so_path.value();

  dlerror(); // Clear any existing error
  m_handle = dlopen(so_path.c_str(), RTLD_LAZY);
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlopen error while loading python-runner library: %s",
           dlsym_error);
    return;
  }

  // Load function pointers
  m_create_runner = (void *(*)())dlsym(m_handle, "python_runner_create");
  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error for python_runner_create: %s", dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_destroy_runner = (void (*)(void *))dlsym(m_handle, "python_runner_destroy");
  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error for python_runner_destroy: %s", dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_add_to_sys_path = (void (*)(void *, const char *))dlsym(
      m_handle, "python_runner_add_to_sys_path");
  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error for python_runner_add_to_sys_path: %s",
           dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_load_module = (void (*)(void *, const char *, const char *))dlsym(
      m_handle, "python_runner_load_module");
  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error for python_runner_load_module: %s",
           dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_forward = (void (*)(void *, const tt::runtime::Tensor *, size_t,
                        const tt::runtime::Device *, tt::runtime::Tensor **,
                        size_t *))dlsym(m_handle, "python_runner_forward");
  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error for python_runner_forward: %s", dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_initialized = true;
}

void *PythonRunnerHandler::createRunner() {
  assert(m_initialized && "PythonRunnerHandler not initialized");
  return m_create_runner();
}

void PythonRunnerHandler::destroyRunner(void *runner) {
  assert(m_initialized && "PythonRunnerHandler not initialized");
  m_destroy_runner(runner);
}

void PythonRunnerHandler::addToSysPath(void *runner, const char *path) {
  assert(m_initialized && "PythonRunnerHandler not initialized");
  m_add_to_sys_path(runner, path);
}

void PythonRunnerHandler::loadModule(void *runner, const char *module_name,
                                     const char *function_name) {
  assert(m_initialized && "PythonRunnerHandler not initialized");
  m_load_module(runner, module_name, function_name);
}

std::vector<tt::runtime::Tensor>
PythonRunnerHandler::forward(void *runner,
                             const std::vector<tt::runtime::Tensor> &inputs,
                             const tt::runtime::Device &device) {
  assert(m_initialized && "PythonRunnerHandler not initialized");

  tt::runtime::Tensor *output_array = nullptr;
  size_t output_count = 0;

  m_forward(runner, inputs.data(), inputs.size(), &device, &output_array,
            &output_count);

  std::vector<tt::runtime::Tensor> outputs;
  if (output_array && output_count > 0) {
    outputs.assign(output_array, output_array + output_count);
    // Note: Assuming the C API allocates memory that we need to free
    // If the API owns the memory, remove this free call
    free(output_array);
  }

  return outputs;
}

// Global singleton instance
static PythonRunnerHandler &getPythonRunnerHandler() {
  static PythonRunnerHandler handler;
  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() { handler.initialize(); });
  return handler;
}

std::unique_ptr<SOLoadedExecutableInstance>
SOLoadedExecutableInstance::createInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  struct make_unique_enabler : public SOLoadedExecutableInstance {
    make_unique_enabler(std::shared_ptr<SOExecutableImage> executable_image,
                        std::vector<DeviceInstance *> &&addressable_devices,
                        ClientInstance *client_instance)
        : SOLoadedExecutableInstance(std::move(executable_image),
                                     std::move(addressable_devices),
                                     client_instance) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices),
                                               client_instance);
}

SOLoadedExecutableInstance::SOLoadedExecutableInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {}

std::shared_ptr<SOExecutableImage>
SOLoadedExecutableInstance::getSharedExecutableImage() const {
  return std::static_pointer_cast<SOExecutableImage>(m_executable_image);
}

void SOLoadedExecutableInstance::releaseResources() {
  if (m_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Release SO-specific resources
  m_executable_image.reset();

  m_deleted = true;
}

tt_pjrt_status
SOLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "SOLoadedExecutableInstance::Execute");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    DLOG_F(ERROR, "Device count mismatch: %zu vs %zu", args->num_devices,
           m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    DLOG_F(ERROR, "Argument count mismatch: %zu vs %zu", args->num_args,
           m_executable_image->getNumInputs());
    return tt_pjrt_status::kInternal;
  }

  std::optional<tt::runtime::Device> runtime_device =
      getOrCreateMeshDevice(args->argument_lists, args->num_args,
                            args->num_devices, args->execute_device);

  if (!runtime_device) {
    // Logging is done inside `getOrCreateMeshDevice`.
    return tt_pjrt_status::kInternal;
  }

  // Assuming only one program per SO for now.
  std::uint32_t program_index = 0;

  std::vector<tt::runtime::Tensor> input_tensors;
  input_tensors.reserve(args->num_args);
  tt_pjrt_status status = getInputRuntimeTensors(
      args->argument_lists, args->num_args, args->num_devices, *runtime_device,
      program_index, input_tensors);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }

  if (m_executable_image->getCompileOptions().export_tensors) {
    dumpInputs(input_tensors);
  }

  CompileOptions options = m_executable_image->getCompileOptions();
  std::string lang =
      options.backend == BackendRuntime::TTNNCodegenPy ? "Python" : "C++";
  std::cout << lang << " codegen successful. Check "
            << options.export_path.value() << " for the results." << std::endl;

  if (options.dry_run) {
    // Dry run mode: skip execution and return zero-initialized output buffers
    createDefaultOutputBuffers(args->output_lists, args->num_devices);
  } else {
    // Execute the generated code using PythonRunnerHandler
    PythonRunnerHandler &handler = getPythonRunnerHandler();

    if (!handler.isInitialized()) {
      DLOG_F(ERROR, "PythonRunnerHandler not initialized - cannot execute SO");
      return tt_pjrt_status::kInternal;
    }

    void *runner = handler.createRunner();
    if (!runner) {
      DLOG_F(ERROR, "Failed to create Python runner instance");
      return tt_pjrt_status::kInternal;
    }

    handler.addToSysPath(runner, options.export_path.value().c_str());
    handler.loadModule(runner, "main", "forward");
    std::vector<tt::runtime::Tensor> output_tensors =
        handler.forward(runner, input_tensors, *runtime_device);

    handler.destroyRunner(runner);

    if (output_tensors.size() != m_executable_image->getNumOutputs()) {
      DLOG_F(ERROR,
             "Runtime produced different number of output tensors (%zu) than "
             "the compiler estimated number of outputs (%zu)",
             output_tensors.size(), m_executable_image->getNumOutputs());
      return tt_pjrt_status::kInternal;
    }

    fillPJRTOutputLists(output_tensors, args->num_devices, args->output_lists,
                        m_executable_image->getOutputTypes());
  }

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      device_complete_event->markAsReady(tt_pjrt_status::kSuccess);

      // Releasing ownership to the PJRT API caller
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

void SOLoadedExecutableInstance::createDefaultOutputBuffers(
    PJRT_Buffer **const *output_lists, size_t num_devices) {
  size_t num_outputs = m_executable_image->getNumOutputs();

  for (size_t device_index = 0; device_index < num_devices; ++device_index) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      std::vector<std::uint32_t> output_shape =
          m_executable_image->getOutputShape(output_index);
      PJRT_Buffer_Type output_type =
          m_executable_image->getOutputTypes()[output_index];
      ::tt::target::DataType runtime_data_type =
          data_type_utils::convertPJRTToRuntimeDataType(output_type);
      std::uint32_t element_size =
          tt::runtime::utils::dataTypeElementSize(runtime_data_type);

      // We create a row-major tensor. Last stride is 1, one before is the last
      // dimension size, etc. That means the right algorithm is the exclusive
      // right scan.
      std::vector<std::uint32_t> strides(output_shape.size());
      std::exclusive_scan(output_shape.rbegin(), output_shape.rend(),
                          strides.rbegin(), std::uint32_t(1),
                          std::multiplies<>());

      tt::runtime::Tensor host_tensor = tt::runtime::createOwnedHostTensor(
          nullptr, output_shape, strides, element_size, runtime_data_type);

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              host_tensor, std::move(output_shape),
              m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              output_type, device_index, m_client_instance);

      output_buffer->markAsDataReady();

      // Release ownership to the PJRT API caller
      output_lists[device_index][output_index] = *output_buffer.release();
    }
  }
}

std::optional<tt::runtime::Tensor>
SOLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device runtime_device, size_t num_devices,
    std::uint32_t program_index, size_t arg_index) {
  // Assert that all buffer instances have the same prepared tensor.
  // NOTE: In case of sharded tensor we have multiple buffer instances on the
  // PJRT side, but on our side (tt-mlir runtime) we prepare a single
  // multi-device tensor.
  assert(!arg_buffers.empty());
  std::optional<tt::runtime::Tensor> prepared_tensor =
      arg_buffers[0]->getPreparedTensor();
  for (size_t i = 1; i < arg_buffers.size(); ++i) {
    assert(arg_buffers[i]->getPreparedTensor().has_value() ==
           prepared_tensor.has_value());
    if (prepared_tensor.has_value()) {
      assert(arg_buffers[i]->getPreparedTensor()->handle ==
             prepared_tensor->handle);
    }
  }

  // For SO path, we don't have layout information from flatbuffer,
  // so we can't check if the prepared tensor has the correct layout.
  // As a hack, we'll always use whatever we have prepared.
  if (prepared_tensor.has_value()) {
    DLOG_F(LOG_DEBUG,
           "Reusing already prepared input tensor for argument index %zu",
           arg_index);
    return *prepared_tensor;
  }

  // We don't have an already prepared tensor so we need to prepare it now.
  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);
  if (mlir::failed(strategy)) {
    DLOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  tt::runtime::Tensor input_tensor =
      getTensorFromStrategy(arg_buffers, *strategy);

  tt::runtime::Tensor laid_out_tensor = convertTensorLayout(
      input_tensor, program_index, arg_index, runtime_device);

  // Right now we don't actually lay out tensors, so no need to save it.

  return laid_out_tensor;
}

tt::runtime::Tensor SOLoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  return input_tensor;
}

} // namespace tt::pjrt

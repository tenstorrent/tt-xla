// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/so_loaded_executable_instance.h"

// c++ standard library includes
#include <iostream>
#include <mutex>
#include <numeric>
#include <unordered_map>

// tt-mlir includes
#include "tools/tt-alchemist/python_runner/python_runner.hpp"
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

std::unique_ptr<SOLoadedExecutableInstance>
SOLoadedExecutableInstance::createInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  DLOG_F(LOG_DEBUG, "SOLoadedExecutableInstance::createInstance");

  return std::unique_ptr<SOLoadedExecutableInstance>(
      new SOLoadedExecutableInstance(std::move(executable_image),
                                     addressable_devices, client_instance));
}

SOLoadedExecutableInstance::SOLoadedExecutableInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {
  DLOG_F(LOG_DEBUG, "SOLoadedExecutableInstance::SOLoadedExecutableInstance");
}

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
    // Execute the generated code using PythonModelRunner
    try {
      tt::alchemist::PythonModelRunner runner;
      runner.addToSysPath(options.export_path.value());
      runner.loadModule("main", "main");
      std::vector<tt::runtime::Tensor> output_tensors =
          runner.forward(input_tensors, *runtime_device);

      if (output_tensors.size() != m_executable_image->getNumOutputs()) {
        DLOG_F(ERROR,
               "Runtime produced different number of output tensors (%zu) than "
               "the compiler estimated number of outputs (%zu)",
               output_tensors.size(), m_executable_image->getNumOutputs());
        return tt_pjrt_status::kInternal;
      }

      fillPJRTOutputLists(output_tensors, args->num_devices, args->output_lists,
                          m_executable_image->getOutputTypes());
    } catch (const std::exception &e) {
      DLOG_F(ERROR, "PythonModelRunner execution failed: %s", e.what());
      return tt_pjrt_status::kInternal;
    }
  }

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Event_Destroy` on the event.
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

std::optional<tt::runtime::Tensor>
SOLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device device, size_t num_devices, std::uint32_t program_index,
    size_t arg_index) {
  // For SO path, if buffer already has a prepared tensor, reuse it.
  // Otherwise, perform strategy-based tensor preparation.
  std::optional<tt::runtime::Tensor> prepared_tensor =
      arg_buffers.front()->getPreparedTensor();
  if (prepared_tensor.has_value()) {
    return prepared_tensor;
  }

  // For SO path, we don't have sharding info from flatbuffer, so use identity
  // strategy
  std::unordered_map<std::string, std::string> strategy;
  strategy["strategy"] = "identity";

  return convertTensorLayout(getTensorFromStrategy(arg_buffers, strategy),
                             program_index, arg_index, device);
}

tt::runtime::Tensor SOLoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  // For SO path, we don't have flatbuffer layout information.
  // Just return the input tensor as-is since codegen code handles layouts.
  return input_tensor;
}

void SOLoadedExecutableInstance::createDefaultOutputBuffers(
    PJRT_Buffer **const *output_lists, size_t num_devices) {
  for (size_t output_index = 0;
       output_index < m_executable_image->getNumOutputs(); output_index++) {
    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      std::vector<std::uint32_t> output_shape =
          m_executable_image->getOutputShape(output_index);

      std::vector<std::int64_t> dims(output_shape.begin(), output_shape.end());

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createInputBufferInstance(
              m_executable_image->getOutputTypes()[output_index], dims.data(),
              dims.size(), m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              m_client_instance);

      DLOG_F(LOG_DEBUG,
             "Created default output buffer at index %zu, device %zu",
             output_index, device_index);

      output_buffer->markAsDataReady();

      // Releasing the ownership to the PJRT API caller
      output_lists[device_index][output_index] = *output_buffer.release();
    }
  }
}

} // namespace tt::pjrt

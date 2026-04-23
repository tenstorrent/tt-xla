// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/so_loaded_executable_instance.h"

// c++ standard library includes
#include <iostream>
#include <mutex>

// tracy includes
#include "tracy/Tracy.hpp"

// tt-mlir includes
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "api/tensor.h"
#include "utils/logging.h"
#include "utils/status.h"

#if TTXLA_ENABLE_EMITPY_EXECUTION
#include <tools/tt-alchemist/python_runner/python_runner.hpp>

constexpr static char MODULE_NAME[] = "main";
constexpr static char ENTRYPOINT_NAME[] = "main_for_test";
#endif

namespace tt::pjrt {

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
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "SOLoadedExecutableInstance::Execute");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    LOG_F(ERROR, "Device count mismatch: %zu vs %zu", args->num_devices,
          m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    LOG_F(ERROR, "Argument count mismatch: %zu vs %zu", args->num_args,
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

  if (options.dry_run || options.backend != BackendRuntime::TTNNCodegenPy) {
    // dry_run mode or non-Python codegen: return zero-filled output buffers.
    tt_pjrt_status status =
        createDefaultOutputBuffers(args->output_lists, args->num_devices);
    if (!tt_pjrt_status_is_ok(status)) {
      return status;
    }
  } else {
#if !TTXLA_ENABLE_EMITPY_EXECUTION
    LOG_F(ERROR, "EmitPy execution requested, but this build does not include "
                 "PythonModelRunner support");
    return tt_pjrt_status::kInternal;
#else
    // Execute the generated Python code via PythonModelRunner.
    tt::alchemist::PythonModelRunner runner;
    runner.addToSysPath(options.export_path.value());
    runner.loadModule(MODULE_NAME, ENTRYPOINT_NAME);

    std::vector<tt::runtime::Tensor> output_tensors =
        runner.forward(input_tensors, *runtime_device);

    if (output_tensors.size() != m_executable_image->getNumOutputs()) {
      LOG_F(ERROR,
            "PythonModelRunner produced different number of output tensors "
            "(%zu) than the compiler estimated number of outputs (%zu)",
            output_tensors.size(), m_executable_image->getNumOutputs());
      return tt_pjrt_status::kInternal;
    }

    tt_pjrt_status status = fillPJRTOutputLists(
        output_tensors, args->num_devices, args->output_lists,
        m_executable_image->getOutputTypes());
    if (!tt_pjrt_status_is_ok(status)) {
      return status;
    }
#endif
  }
  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      EventInstance::markAsReadyAndCallback(device_complete_event.get(),
                                            tt_pjrt_status::kSuccess);

      // Releasing ownership to the PJRT API caller
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

std::optional<tt::runtime::Tensor>
SOLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device runtime_device, size_t num_devices,
    std::uint32_t program_index, size_t arg_index) {
  ZoneScoped;

  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);

  if (mlir::failed(strategy)) {
    LOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  PjrtTensor &tensor = PjrtTensor::from_pjrt_buffers(
      arg_buffers, m_executable_image->getDevicesMeshShape(), *strategy);

  return tensor.runtime_tensor();
}

} // namespace tt::pjrt

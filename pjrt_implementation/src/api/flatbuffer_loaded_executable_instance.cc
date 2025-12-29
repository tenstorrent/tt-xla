// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/flatbuffer_loaded_executable_instance.h"

// c++ standard library includes
#include <cassert>

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/error_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "utils/logging.h"
#include "utils/utils.h"

namespace tt::pjrt {

std::unique_ptr<FlatbufferLoadedExecutableInstance>
FlatbufferLoadedExecutableInstance::createInstance(
    std::shared_ptr<FlatbufferExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  struct make_unique_enabler : public FlatbufferLoadedExecutableInstance {
    make_unique_enabler(
        std::shared_ptr<FlatbufferExecutableImage> executable_image,
        std::vector<DeviceInstance *> &&addressable_devices,
        ClientInstance *client_instance)
        : FlatbufferLoadedExecutableInstance(std::move(executable_image),
                                             std::move(addressable_devices),
                                             client_instance) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices),
                                               client_instance);
}

FlatbufferLoadedExecutableInstance::FlatbufferLoadedExecutableInstance(
    std::shared_ptr<FlatbufferExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {}

std::optional<tt::runtime::Tensor>
FlatbufferLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device runtime_device, size_t num_devices,
    std::uint32_t program_index, size_t arg_index) {

  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  tt::runtime::Layout expected_layout = tt::runtime::getLayout(
      executable_image->getFlatbufferBinary(), program_index, arg_index);

  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);
  if (mlir::failed(strategy)) {
    DLOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  PjrtTensor &tensor =
      PjrtTensor::init(arg_buffers, runtime_device, expected_layout,
                       m_executable_image->getDevicesMeshShape(), *strategy);

  return tensor.device_tensor();
}

void FlatbufferLoadedExecutableInstance::fillPJRTOutputLists(
    const std::vector<tt::runtime::Tensor> &output_tensors,
    const tt::runtime::Device &device, size_t num_devices,
    PJRT_Buffer **const *output_lists,
    const std::vector<PJRT_Buffer_Type> &expected_output_data_types) {
  size_t n_prog_output_tensors = output_tensors.size();

  // Iterate over the available tensors and devices, filling in the PJRT Buffer
  // outputs. The output BufferInstance is initialized with a device tensor
  // which is lazily returned to host when CopyToHost is called.
  for (size_t output_index = 0; output_index < n_prog_output_tensors;
       output_index++) {
    tt::runtime::Tensor outputDeviceTensor = output_tensors[output_index];

    std::vector<BufferInstance *> shards;
    shards.reserve(num_devices);

    for (int device_index = 0; device_index < num_devices; ++device_index) {
      std::vector<std::uint32_t> output_shape = getOutputShape(output_index);

      // For a given output_index, each device will get the same
      // outputDeviceTensor This is trivially correct for replicated outputs.
      // For sharded outputs, the outputDeviceTensor is a multi-device tensor
      // and represents all shards. Therefore, the outputDevice tensor will
      // still be the same for each device, and the BufferInstance will store
      // which shard to retrieve via the device index, when requested in
      // CopyToHost.
      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              std::move(output_shape), m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              expected_output_data_types[output_index], device_index);
      DLOG_F(LOG_DEBUG,
             "Filled output at output_index %zu device_index %d with shape %s "
             "and UID %zu",
             output_index, device_index, output_buffer->toShapeStr().c_str(),
             output_buffer->getUID());

      output_buffer->markAsDataReady();

      shards.emplace_back(output_buffer.get());

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Buffer_Destroy` on the buffer.
      output_lists[device_index][output_index] = *output_buffer.release();
    }

    PjrtTensor::init(outputDeviceTensor, shards, device);
  }
}

std::vector<std::uint32_t>
FlatbufferLoadedExecutableInstance::getOutputShape(size_t output_index) {
  std::vector<std::uint32_t> outputShape =
      m_executable_image->getOutputShape(output_index);
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      m_executable_image->getOutputSharding(output_index);

  if (outputSharding.getShardType() ==
          mlir::tt::ttcore::MeshShardType::Identity ||
      outputSharding.getShardType() ==
          mlir::tt::ttcore::MeshShardType::Replicate) {
    return outputShape;
  }
  llvm::SmallVector<int64_t> output_sharding_shard_shape =
      outputSharding.getShardShape();
  assert(output_sharding_shard_shape.size() == outputShape.size() &&
         "Output sharding shape doesn't match the output shape");

  for (size_t i = 0; i < outputShape.size(); ++i) {
    assert(outputShape[i] % output_sharding_shard_shape[i] == 0 &&
           "Output shape is not divisible by the sharding shape");
    outputShape[i] /= output_sharding_shard_shape[i];
  }

  return outputShape;
}

std::shared_ptr<FlatbufferExecutableImage>
FlatbufferLoadedExecutableInstance::getSharedExecutableImage() const {
  return std::static_pointer_cast<FlatbufferExecutableImage>(
      m_executable_image);
}

void FlatbufferLoadedExecutableInstance::releaseResources() {
  if (m_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Here we should drop executable's reference to the internal runtime object
  // and associated resources, but we currently store no runtime objects so
  // releasing only resources.
  m_executable_image.reset();

  m_deleted = true;
}

// TODO(mrakita): Make this method work in asynchronous fashion.
tt_pjrt_status FlatbufferLoadedExecutableInstance::execute(
    PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "FlatbufferLoadedExecutableInstance::Execute");
  LOG_BRINGUP_STAGE("RUNTIME_EXECUTION_START");

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

  // Assuming only one program per flatbuffer for now.
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

  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  auto r = utils::invoke_noexcept(tt::runtime::submit, *runtime_device,
                                  executable_image->getFlatbufferBinary(),
                                  program_index, input_tensors);

  if (!r) {
    m_client_instance->closeMeshDevice();
    return tt_pjrt_status::kInternal;
  }

  std::vector<tt::runtime::Tensor> &output_tensors = *r;

  if (output_tensors.size() != m_executable_image->getNumOutputs()) {
    DLOG_F(ERROR,
           "Runtime produced different number of output tensors (%zu) than the "
           "compiler estimated number of outputs (%zu)",
           output_tensors.size(), m_executable_image->getNumOutputs());
    return tt_pjrt_status::kInternal;
  }

  fillPJRTOutputLists(output_tensors, *runtime_device, args->num_devices,
                      args->output_lists, m_executable_image->getOutputTypes());

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      device_complete_event->markAsReady(tt_pjrt_status::kSuccess);

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Event_Destroy` on the event.
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/flatbuffer_loaded_executable_instance.h"

// c++ standard library includes
#include <cassert>
#include <filesystem>
#include <numeric>

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/error_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "utils/logging.h"

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

std::optional<tt::runtime::Device>
FlatbufferLoadedExecutableInstance::getOrCreateMeshDevice(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, PJRT_Device *pjrt_device) {
  std::unordered_set<int> device_ids =
      getDeviceIds(argument_lists, num_args, num_devices);

  const std::vector<std::uint32_t> &devices_mesh_shape =
      m_executable_image->getDevicesMeshShape();
  size_t mesh_shape_num_devices = static_cast<size_t>(
      std::accumulate(devices_mesh_shape.begin(), devices_mesh_shape.end(), 1,
                      std::multiplies<std::uint32_t>{}));

  if (device_ids.size() != mesh_shape_num_devices) {
    DLOG_F(ERROR,
           "Input buffers are placed on a different number of devices (%zu) "
           "than in the mesh shape estimated by the compiler (%zu)",
           device_ids.size(), mesh_shape_num_devices);
    return std::nullopt;
  }

  DeviceInstance *device_instance = DeviceInstance::unwrap(pjrt_device);
  if (device_instance &&
      !(device_ids.size() == 1 &&
        *device_ids.begin() == device_instance->getGlobalDeviceId())) {
    DLOG_F(ERROR, "Input buffers are placed on a different device than the one "
                  "specified in the execute_device argument");
    return std::nullopt;
  }

  // TODO(mrakita): Currently runtime doesn't allow us to open specific devices
  // subset, we can only open contiguous subset of devices starting from some
  // offset. We need to keep track of opened devices in Client and map the
  // buffers devices to these devices.
  // https://github.com/tenstorrent/tt-xla/issues/502

  return m_client_instance->getOrCreateMeshDevice(devices_mesh_shape);
}

std::unordered_set<int> FlatbufferLoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices) {
  std::unordered_set<int> device_ids;

  for (size_t device_index = 0; num_args && device_index < num_devices;
       device_index++) {
    const BufferInstance *buffer =
        BufferInstance::unwrap(argument_lists[device_index][0]);
    int64_t buffer_device_id = buffer->getDevice()->getGlobalDeviceId();
    device_ids.emplace(buffer_device_id);
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit. Maybe use `execute_device` from the args?
  if (device_ids.size() == 0) {
    assert(!m_addressable_devices.empty());
    device_ids.emplace(m_addressable_devices.front()->getGlobalDeviceId());
  }

  return device_ids;
}

tt_pjrt_status FlatbufferLoadedExecutableInstance::getInputRuntimeTensors(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, const tt::runtime::Device &runtime_device,
    std::uint32_t program_index,
    std::vector<tt::runtime::Tensor> &input_tensors) {
  for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
    std::vector<BufferInstance *> arg_buffers;
    arg_buffers.reserve(num_devices);

    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      BufferInstance *buffer =
          BufferInstance::unwrap(argument_lists[device_index][arg_index]);
      arg_buffers.push_back(buffer);
    }

    std::optional<tt::runtime::Tensor> prepared_tensor = prepareInputTensor(
        arg_buffers, runtime_device, num_devices, program_index, arg_index);

    if (!prepared_tensor.has_value()) {
      // Error is reported in `prepareInputTensor`.
      return tt_pjrt_status::kInternal;
    }

    input_tensors.push_back(*prepared_tensor);
  }

  return tt_pjrt_status::kSuccess;
}

std::optional<tt::runtime::Tensor>
FlatbufferLoadedExecutableInstance::prepareInputTensor(
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

  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  // Check if we already have a prepared tensor corresponding to the buffer
  // instance(s); i.e. a tensor with the layout which this executable is
  // expecting. If so, we can just reuse this tensor.
  tt::runtime::Layout expected_layout = tt::runtime::getLayout(
      executable_image->getFlatbufferBinary(), program_index, arg_index);
  if (prepared_tensor.has_value() &&
      tt::runtime::hasLayout(*prepared_tensor, expected_layout)) {
    DLOG_F(LOG_DEBUG,
           "Reusing already prepared input tensor for argument index %zu",
           arg_index);
    return *prepared_tensor;
  }

  // We don't have an already prepared tensor so we need to prepare it now.
  // This involves two steps:
  // 1) Create a multi-device tensor from the input buffer instances
  //   according to the sharding strategy (if needed).
  // 2) Convert the layout of the tensor to the layout expected by the
  //  executable.
  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      FlatbufferLoadedExecutableInstance::fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);
  if (mlir::failed(strategy)) {
    DLOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  tt::runtime::Tensor input_tensor =
      getTensorFromStrategy(arg_buffers, *strategy);

  tt::runtime::Tensor laid_out_tensor = convertTensorLayout(
      input_tensor, program_index, arg_index, runtime_device);

  // Save the prepared tensor (properly laid out tensor) inside of the buffer
  // instance(s), so we can reuse it on subsequent executions of the same
  // executable.
  for (size_t i = 0; i < arg_buffers.size(); ++i) {
    arg_buffers[i]->setPreparedTensor(laid_out_tensor);
  }

  return laid_out_tensor;
}

mlir::FailureOr<std::unordered_map<std::string, std::string>>
FlatbufferLoadedExecutableInstance::fillStrategyMapFromSharding(
    const mlir::tt::sharding_utils::MeshSharding &meshSharding,
    size_t num_devices) {
  std::unordered_map<std::string, std::string> strategy;
  mlir::tt::ttcore::MeshShardType meshType = meshSharding.getShardType();
  if (meshType == mlir::tt::ttcore::MeshShardType::Replicate) {
    // If there is only one device, the output will be replicated, but there is
    // no need to replicate.
    if (num_devices == 1) {
      strategy["strategy"] = "identity";
    } else {
      strategy["strategy"] = "replicate";
      strategy["replication_factor"] = std::to_string(num_devices);
    }
  } else if (meshType == mlir::tt::ttcore::MeshShardType::Devices) {
    llvm::SmallVector<int64_t> mesh_shape_data = meshSharding.getMeshShape();
    assert(mesh_shape_data.size() <= 2 && mesh_shape_data.size() >= 1);
    if (mesh_shape_data.size() == 1) {
      strategy["strategy"] = "shard";
      strategy["shard_dim"] = std::to_string(mesh_shape_data[0]);
    }
    if (mesh_shape_data.size() == 2) {
      strategy["strategy"] = "shard_2d";
      strategy["mesh_shape_y"] = std::to_string(mesh_shape_data[0]);
      strategy["mesh_shape_x"] = std::to_string(mesh_shape_data[1]);
    }
  } else if (meshType == mlir::tt::ttcore::MeshShardType::Identity) {
    strategy["strategy"] = "identity";
  } else {
    return mlir::failure();
  }
  return strategy;
}

// TODO: We are using std::maps with strings as that is the way it is defined in
// the tt::runtime, instead of a more structured approach with structs and/or
// enums. See issue: https://github.com/tenstorrent/tt-mlir/issues/2513
tt::runtime::Tensor FlatbufferLoadedExecutableInstance::getTensorFromStrategy(
    const std::vector<BufferInstance *> &arg_buffers,
    const std::unordered_map<std::string, std::string> &strategy) {
  if (strategy.at("strategy") == "identity") {
    std::optional<tt::runtime::Tensor> host_runtime_tensor =
        arg_buffers.front()->getHostRuntimeTensor();
    assert(
        host_runtime_tensor.has_value() &&
        "Host tensor should be available in the buffer instance at this point");
    return *host_runtime_tensor;
  }

  std::vector<tt::runtime::Tensor> runtime_tensor_shards;
  runtime_tensor_shards.reserve(arg_buffers.size());
  for (const BufferInstance *buffer : arg_buffers) {
    std::optional<tt::runtime::Tensor> host_runtime_tensor =
        buffer->getHostRuntimeTensor();
    assert(
        host_runtime_tensor.has_value() &&
        "Host tensor should be available in the buffer instance at this point");
    runtime_tensor_shards.push_back(*host_runtime_tensor);
  }

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      runtime_tensor_shards, strategy,
      m_executable_image->getDevicesMeshShape());
  tt::runtime::setTensorRetain(tensor, /*retain=*/true);

  return tensor;
}

tt::runtime::Tensor FlatbufferLoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  tt::runtime::Layout layout = tt::runtime::getLayout(
      executable_image->getFlatbufferBinary(), program_index, arg_index);

  return tt::runtime::toLayout(input_tensor, runtime_device, layout,
                               tt::runtime::getTensorRetain(input_tensor));
}

tt_pjrt_status FlatbufferLoadedExecutableInstance::untilizeToHost(
    const std::vector<tt::runtime::Tensor> &output_tensors, size_t num_devices,
    std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors) {
  for (size_t output_index = 0; output_index < output_tensors.size();
       ++output_index) {
    std::vector<tt::runtime::Tensor> untilized_output =
        tt::runtime::toHost(output_tensors[output_index], /* untilize */ true);

    // If the output is a replicated scalar or tensor, we expect only one tensor
    // on output, so we need to fill the rest of the output tensors with the
    // same tensors, to match the number of devices.
    if (untilized_output.size() != num_devices) {
      // If the size of the output is not 1 nor num_devices, we have an error.
      if (untilized_output.size() > 1) {
        DLOG_F(ERROR,
               "Untilize to host produced invalid number of output tensors: "
               "expected %zu, got %zu",
               num_devices, untilized_output.size());
        return tt_pjrt_status::kInternal;
      }
      for (size_t device_index = 1; device_index < num_devices;
           ++device_index) {
        untilized_output.emplace_back(untilized_output[0]);
      }
    }

    untilized_output_tensors.emplace_back(std::move(untilized_output));
  }

  return tt_pjrt_status::kSuccess;
}

void FlatbufferLoadedExecutableInstance::fillPJRTOutputLists(
    const std::vector<std::vector<tt::runtime::Tensor>>
        &untilized_output_tensors,
    size_t num_devices, PJRT_Buffer **const *output_lists,
    const std::vector<PJRT_Buffer_Type> &expected_output_data_types) {
  size_t num_outputs = untilized_output_tensors.size();

  for (int device_index = 0; device_index < num_devices; ++device_index) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      tt::runtime::Tensor output_tensor =
          untilized_output_tensors[output_index][device_index];
      std::vector<std::uint32_t> output_shape = getOutputShape(output_index);

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              output_tensor, std::move(output_shape),
              m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              expected_output_data_types[output_index]);

      output_buffer->markAsDataReady();

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Buffer_Destroy` on the buffer.
      output_lists[device_index][output_index] = *output_buffer.release();
    }
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

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    DLOG_F(ERROR,
           "Requested number of devices to run the executable on (%zu) doesn't "
           "match the compiler estimated number of devices (%zu)",
           args->num_devices, m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    DLOG_F(ERROR,
           "Requested number of arguments to provide to the executable (%zu) "
           "doesn't match the compiler estimated number of inputs (%zu)",
           args->num_args, m_executable_image->getNumInputs());
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

  if (m_executable_image->getCompileOptions().dump_inputs) {
    dumpInputs(input_tensors);
  }

  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  std::vector<tt::runtime::Tensor> output_tensors = tt::runtime::submit(
      *runtime_device, executable_image->getFlatbufferBinary(), program_index,
      input_tensors);

  if (output_tensors.size() != m_executable_image->getNumOutputs()) {
    DLOG_F(ERROR,
           "Runtime produced different number of output tensors (%zu) than the "
           "compiler estimated number of outputs (%zu)",
           output_tensors.size(), m_executable_image->getNumOutputs());
    return tt_pjrt_status::kInternal;
  }

  // [James] TODO fill PJRT output lists with device tensors, inserting into prepared_tensor

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

// This should ideally live in the base class, and dumping should be exposed via
// both paths. As the .so execution is not yet implemented, as a hack this is
// implemented here for now.
void FlatbufferLoadedExecutableInstance::dumpInputs(
    const std::vector<tt::runtime::Tensor> &input_tensors) {
  DLOG_F(LOG_DEBUG, "FlatbufferLoadedExecutableInstance::dumpInputs");

  assert(m_executable_image->getCompileOptions().export_path.has_value() &&
         "Export path must be set when dumping inputs");

  std::filesystem::path dump_dir =
      std::filesystem::path(
          m_executable_image->getCompileOptions().export_path.value()) /
      "input_tensors";
  std::filesystem::create_directories(dump_dir);

  for (int i = 0; i < input_tensors.size(); ++i) {
    std::string filename = "arg" + std::to_string(i) + ".tensorbin";
    std::filesystem::path filepath = dump_dir / filename;

    tt::runtime::dumpTensor(input_tensors[i], filepath.string());
  }
}

} // namespace tt::pjrt

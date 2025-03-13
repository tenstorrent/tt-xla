// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/loaded_executable_instance.h"

#include <string>
#include <unordered_map>

// tt-mlir includes
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {

LoadedExecutableInstance::~LoadedExecutableInstance() { image_->DecRef(); }

void LoadedExecutableInstance::BindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy =
      +[](PJRT_LoadedExecutable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");
    delete LoadedExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(
        LOG_DEBUG,
        "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");
    LoadedExecutableInstance *loaded_executable =
        LoadedExecutableInstance::Unwrap(args->executable);
    // TODO: Set addressable devices in the loaded executable class to only the
    // devices being utilized, rather than all addressable devices. This way,
    // the number of devices will be determined by the list length instead of a
    // separate field in the class.
    const std::vector<DeviceInstance *> &addressable_devices =
        loaded_executable->addressable_devices();
    int num_addressable_devices =
        loaded_executable->get_num_devices_to_utilize();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(addressable_devices.data()));
    args->num_addressable_devices = num_addressable_devices;
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");
    return ErrorInstance::MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->Execute(args));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");
    LoadedExecutableInstance *loaded_exe =
        LoadedExecutableInstance::Unwrap(args->loaded_executable);
    ExecutableImage *image = loaded_exe->image_;

    image->AddRef();
    args->executable = *image;
    return nullptr;
  };
}

tt_pjrt_status
LoadedExecutableInstance::Execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  // Check that the number of devices matches the number of devices counted
  // from the VHLO module.
  size_t num_devices = args->num_devices;
  assert(num_devices == num_devices_to_utilize_);

  const tt::runtime::Binary &binary = image_->get_binary();

  std::vector<tt::runtime::Tensor> rt_inputs;
  for (size_t i = 0; i < args->num_args; ++i) {
    std::vector<std::shared_ptr<void>> data;
    BufferInstance *buffer;
    for (size_t j = 0; j < num_devices; ++j) {
      buffer = BufferInstance::Unwrap(args->argument_lists[j][i]);
      data.push_back(buffer->get_host_buffer_ptr());
    }
    std::unordered_map<std::string, std::string> strategy =
        getStrategyMapFromSharding(image_->getInputSharding(i), num_devices);
    rt_inputs.push_back(getTensorFromStrategy(strategy, buffer, data));
  }

  std::vector<int> device_ids = getDeviceIds(
      args->argument_lists, addressable_devices_, args->num_args, num_devices);

  tt::runtime::Device device = tt::runtime::openDevice(device_ids);
  std::vector<tt::runtime::Tensor> input_tensors;
  int size_inputs = rt_inputs.size();
  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, 0, rt_inputs);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0);
  std::vector<std::vector<tt::runtime::Tensor>> rt_outputs_list(num_devices);

  for (size_t i = 0; i < output_specs.size(); ++i) {
    std::vector<tt::runtime::Tensor> untilized_output =
        tt::runtime::toHostShardAware(rt_outputs[i], true);
    for (size_t j = 0; j < untilized_output.size(); j++) {
      rt_outputs_list[j].push_back(untilized_output[j]);
    }
  }

  fillPJRTOutputLists(rt_outputs_list, output_specs, num_devices,
                      args->output_lists);

  for (size_t i = 0; i < rt_outputs.size(); ++i) {
    tt::runtime::deallocateTensor(rt_outputs[i], /*force=*/true);
  }

  if (args->device_complete_events) {
    for (int i = 0; i < num_devices; i++)
      args->device_complete_events[i] = *(new EventInstance());
  }

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

std::unordered_map<std::string, std::string>
LoadedExecutableInstance::getStrategyMapFromSharding(
    const mlir::tt::sharding_utils::MeshSharding &meshSharding,
    size_t num_devices) {
  mlir::tt::MeshShardType meshType = meshSharding.getShardType();
  std::unordered_map<std::string, std::string> strategy;
  if (meshType == mlir::tt::MeshShardType::Replicate) {
    if (num_devices == 1) {
      strategy["strategy"] = "manual";
    } else {
      strategy["strategy"] = "replicate";
      strategy["replication_factor"] = std::to_string(num_devices);
    }
  } else if (meshType == mlir::tt::MeshShardType::Devices) {
    llvm::ArrayRef<int64_t> shardShape = meshSharding.getShardShape();
    if (shardShape.size() == 2) {
      strategy["strategy"] = "shard_2d";
      strategy["mesh_shape_y"] = std::to_string(shardShape[0]);
      strategy["mesh_shape_x"] = std::to_string(shardShape[1]);
    } else if (shardShape.size() == 1) {
      strategy["strategy"] = "shard";
      strategy["shard_dim"] = "0";
    } else {
      DLOG_F(ERROR, "Invalid mesh sharding type");
      return strategy;
    }
  } else if (meshType == mlir::tt::MeshShardType::Manual) {
    strategy["strategy"] = "manual";
  } else {
    DLOG_F(ERROR, "Invalid mesh sharding type");
    return strategy;
  }
  return strategy;
}

tt::runtime::Tensor LoadedExecutableInstance::getTensorFromStrategy(
    const std::unordered_map<std::string, std::string> &strategy,
    BufferInstance *buffer, std::vector<std::shared_ptr<void>> &data) {
  if (strategy.at("strategy") == "manual") {
    return buffer->getTensor();
  }
  std::pair<tt::target::DataType, size_t> tt_buffer_type =
      buffer->get_tt_buffer_type();
  tt::runtime::TensorDesc tensor_desc = {
      buffer->get_shape(), buffer->get_stride(),
      static_cast<std::uint32_t>(tt_buffer_type.second), tt_buffer_type.first};
  return tt::runtime::createTensor(data, tensor_desc, strategy);
}

std::vector<std::uint32_t>
LoadedExecutableInstance::getOuputShape(size_t index, size_t num_devices) {
  std::vector<std::uint32_t> outputShape = image_->get_output_shape(index);
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      image_->getOutputSharding(index);
  std::unordered_map<std::string, std::string> shardingStrategy =
      getStrategyMapFromSharding(outputSharding, num_devices);
  if (shardingStrategy.at("strategy") == "shard") {
    outputShape[0] = outputShape[0] / num_devices;
  } else if (shardingStrategy.at("strategy") == "shard_2d") {
    assert(outputShape[0] % std::stoi(shardingStrategy.at("mesh_shape_y")) ==
           0);
    assert(outputShape[1] % std::stoi(shardingStrategy.at("mesh_shape_x")) ==
           0);
    outputShape[0] =
        outputShape[0] / std::stoi(shardingStrategy.at("mesh_shape_y"));
    outputShape[1] =
        outputShape[1] / std::stoi(shardingStrategy.at("mesh_shape_x"));
  }
  return outputShape;
}

bool LoadedExecutableInstance::isOutputReplicated(size_t index) {
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      image_->getOutputSharding(index);
  mlir::tt::MeshShardType meshType = outputSharding.getShardType();
  return meshType == mlir::tt::MeshShardType::Replicate;
}

void LoadedExecutableInstance::fillPJRTOutputLists(
    std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs_list,
    const std::vector<tt::runtime::TensorDesc> &output_specs,
    size_t num_devices, PJRT_Buffer **const *output_lists) {
  std::vector<bool> is_replicated;
  size_t num_outputs = rt_outputs_list[0].size();
  for (size_t i = 0; i < rt_outputs_list.size(); i++) {
    num_outputs = std::max(num_outputs, rt_outputs_list[i].size());
  }
  for (size_t i = 0; i < output_specs.size(); ++i) {
    is_replicated.push_back(isOutputReplicated(i));
  }
  for (int k = 0; k < num_devices; k++) {
    for (size_t i = 0; i < num_outputs; ++i) {
      // If the output is replicated, we just put the same tensor in all the
      // devices.
      tt::runtime::Tensor output_tensor =
          isOutputReplicated(i) ? rt_outputs_list[0][i] : rt_outputs_list[k][i];
      std::vector<std::uint32_t> output_shape = getOuputShape(i, num_devices);
      std::pair<tt::target::DataType, size_t> type_pair = {
          output_specs[i].dataType, output_specs[i].itemsize};
      auto result_buffer = std::make_unique<BufferInstance>(
          *this->addressable_devices_[k], output_tensor, output_shape,
          output_specs[i].stride, type_pair);
      result_buffer->setType(tt::pjrt::utils::convertElementTypeToBufferType(
          output_specs[i].dataType));
      DLOG_F(DEBUG, "Runtime output id: %d", result_buffer->unique_id());
      output_lists[k][i] = *(result_buffer.release());
    }
  }
}

std::vector<int> LoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists,
    const std::vector<DeviceInstance *> &addressable_devices, size_t num_args,
    size_t num_devices) {
  std::vector<int> device_ids;

  for (size_t i = 0; num_args && i < num_devices; i++) {
    const BufferInstance *buffer = BufferInstance::Unwrap(argument_lists[i][0]);
    int64_t buffer_device_id =
        buffer->device().device_description()->getDeviceId();
    device_ids.push_back(buffer_device_id);
    DLOG_F(DEBUG, "Runtime input id: %d", buffer->unique_id());
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit.
  if (device_ids.size() == 0) {
    device_ids.push_back(
        addressable_devices_[0]->device_description()->getDeviceId());
  }

  return device_ids;
}

} // namespace tt::pjrt

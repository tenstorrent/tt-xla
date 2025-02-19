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

#include <unordered_set>
#include <iostream>

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/utils.h"
#include "tt/runtime/runtime.h"

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
    auto &addressable_devices =
        LoadedExecutableInstance::Unwrap(args->executable)
            ->addressable_devices();
    int num_addressable_devices =
        LoadedExecutableInstance::Unwrap(args->executable)
            ->image_->get_num_addresible_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(addressable_devices.data()));
    args->num_addressable_devices = num_addressable_devices;
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted_Args");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
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

  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  size_t num_devices = args->num_devices;

  tt::runtime::Binary binary(image_->get_binary());

  std::vector<tt::runtime::Tensor> rt_inputs;

  std::unordered_set<int> device_ids;

  for (size_t i = 0; i < num_devices; i++) {
    BufferInstance *buffer =
      BufferInstance::Unwrap(args->argument_lists[i][0]);
    device_ids.insert(
      chip_ids[buffer->device().device_description()->getDeviceId()]);
  }

  for (size_t i = 0; i < args->num_args; ++i) {
    BufferInstance *buffer =
        BufferInstance::Unwrap(args->argument_lists[0][i]);
    rt_inputs.push_back(buffer->getTensor());
    DLOG_F(INFO, "Runtime input id: %d", buffer->unique_id());
  }

  assert(device_ids.size() == num_devices);

  std::vector<int> device_ids_vector(device_ids.begin(), device_ids.end());

  tt::runtime::Device device = tt::runtime::openDevice(device_ids_vector);
  std::vector<tt::runtime::Tensor> input_tensors;
  int size_inputs = rt_inputs.size();
  std::vector<tt::runtime::Tensor> rt_outputs = tt::runtime::submit(device, binary, 0, rt_inputs);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0);

  for (int k=0;k<num_devices;k++)
  {
    for (size_t i = 0; i < output_specs.size(); ++i) {
      bool is_scalar = client_.isOutputScalar(i);
      // PJRT expects an empty shape for scalars.
      std::vector<std::uint32_t> output_shape =
          is_scalar ? std::vector<std::uint32_t>() : output_specs[i].shape;
      auto result_buffer = std::make_unique<BufferInstance>(
          *this->addressable_devices_[k], rt_outputs[i], output_shape,
          output_specs[i].stride);
      result_buffer->setType(tt::pjrt::utils::convertElementTypeToBufferType(
          output_specs[i].dataType));
      DLOG_F(INFO, "Runtime output id: %d", result_buffer->unique_id());
      args->output_lists[k][i] = *(result_buffer.release());
    }
  }

  if (args->device_complete_events) {
    for (int i=0;i<2;i++) args->device_complete_events[i] = *(new EventInstance());
  }

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt

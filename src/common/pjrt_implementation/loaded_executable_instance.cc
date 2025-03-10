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
  assert(args->num_devices == num_devices_to_utilize_);

  int dev_index = 0;
  const tt::runtime::Binary &binary = image_->get_binary();

  std::vector<tt::runtime::Tensor> rt_inputs;
  rt_inputs.reserve(args->num_args);

  std::unordered_set<int> device_ids;

  for (size_t i = 0; i < args->num_args; ++i) {
    BufferInstance *buffer =
        BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
    rt_inputs.emplace_back(buffer->getTensor());
    int64_t buffer_device_id =
        buffer->device().device_description()->getDeviceId();
    device_ids.insert(buffer_device_id);
    DLOG_F(INFO, "Runtime input id: %d", buffer->unique_id());
  }

  std::vector<int> device_ids_vector(device_ids.begin(), device_ids.end());

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit.
  if (device_ids.size() == 0) {
    device_ids_vector.push_back(
        addressable_devices_[0]->device_description()->getDeviceId());
  }

  tt::runtime::Device device = tt::runtime::openDevice(device_ids_vector);

  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, 0, rt_inputs);

  assert(rt_outputs.size() == image_->get_num_outputs());

  for (size_t i = 0; i < image_->get_num_outputs(); ++i) {

    tt::runtime::Tensor untilized_output_tensor =
        tt::runtime::toHost(rt_outputs[i], /*untilize=*/true);
    auto result_buffer = std::make_unique<BufferInstance>(
        *this->addressable_devices_[dev_index], untilized_output_tensor,
        image_->get_output_shape(i), image_->get_output_stride(i));
    tt::runtime::deallocateTensor(rt_outputs[i], /*force=*/true);

    result_buffer->setType(image_->get_output_types()[i]);

    DLOG_F(INFO, "Runtime output id: %d", result_buffer->unique_id());
    args->output_lists[dev_index][i] = *(result_buffer.release());
  }

  if (args->device_complete_events) {
    args->device_complete_events[dev_index] = *(new EventInstance());
  }

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt

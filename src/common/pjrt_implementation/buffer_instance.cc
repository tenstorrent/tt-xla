// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/buffer_instance.h"

#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {
int BufferInstance::id_counter_ = 0;

BufferInstance::~BufferInstance() = default;

BufferInstance::BufferInstance(DeviceInstance &device,
                               tt::runtime::Tensor &tensor,
                               const std::vector<std::uint32_t> &shape,
                               const std::vector<std::uint32_t> &stride)
    : BufferInstance(device, tensor, shape, stride, nullptr) {}

BufferInstance::BufferInstance(DeviceInstance &device,
                               tt::runtime::Tensor &tensor,
                               const std::vector<std::uint32_t> &shape,
                               const std::vector<std::uint32_t> &stride,
                               std::shared_ptr<void> host_buffer_ptr)
    : device_(device), tensor_(tensor), host_buffer_ptr_(host_buffer_ptr) {
  DLOG_F(LOG_DEBUG, "BufferInstance::BufferInstance");
  dims_.resize(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    dims_[i] = shape[i];
  }
  stride_ = stride;
  unique_id_ = id_counter_++;
}

void BufferInstance::ComputeLayout() {
  DLOG_F(LOG_DEBUG, "BufferInstance::ComputeLayout");
}

void BufferInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "BufferInstance::BindApi");
  api->PJRT_Buffer_Destroy =
      +[](PJRT_Buffer_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Destroy");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    delete buffer;
    return nullptr;
  };
  api->PJRT_Buffer_ElementType =
      +[](PJRT_Buffer_ElementType_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ElementType");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    std::optional<PJRT_Buffer_Type> type = buffer->getType();
    if (type.has_value())
      args->type = *type;
    else {
      args->type = buffer->getRuntimeType();
    }
    return nullptr;
  };
  api->PJRT_Buffer_Dimensions =
      +[](PJRT_Buffer_Dimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Dimensions");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_UnpaddedDimensions =
      +[](PJRT_Buffer_UnpaddedDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->unpadded_dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ToHostBuffer");
    BufferInstance *buffer = BufferInstance::Unwrap(args->src);
    if (!args->dst) {
      // Size query.
      return ErrorInstance::MakeError(
          buffer->GetHostSizeInBytes(&args->dst_size));
    } else {
      // Initiate transfer.
      return ErrorInstance::MakeError(
          buffer->CopyToHost(args->dst, args->dst_size,
                             reinterpret_cast<EventInstance **>(&args->event)));
    }
  };
  api->PJRT_Buffer_OnDeviceSizeInBytes =
      +[](PJRT_Buffer_OnDeviceSizeInBytes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_OnDeviceSizeInBytes");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    return nullptr;
  };
  api->PJRT_Buffer_Delete = +[](PJRT_Buffer_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Delete");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    buffer->Delete();
    return nullptr;
  };
  api->PJRT_Buffer_IsDeleted =
      +[](PJRT_Buffer_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsDeleted");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->is_deleted = buffer->is_deleted();
    return nullptr;
  };
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_CopyToDevice");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsOnCpu");
    args->is_on_cpu = BufferInstance::Unwrap(args->buffer)->is_on_cpu();
    return nullptr;
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Device");
    args->device = BufferInstance::Unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_Memory = +[](PJRT_Buffer_Memory_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Memory");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Buffer_ReadyEvent =
      +[](PJRT_Buffer_ReadyEvent_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    std::unique_ptr<EventInstance> onReadyEvent =
        std::make_unique<EventInstance>();
    buffer->on_ready_event_ = onReadyEvent.get();
    // Releasing the ownership to the PJRT API caller since the caller is
    // responsible for calling PJRT_Event_Destroy on event.
    args->event = *onReadyEvent.release();
    return nullptr;
  };
  // TODO: Rework the API to be Aliases(b1, b2) to let the plugin explicitly
  // check for aliases.
  api->PJRT_Buffer_GetMemoryLayout =
      +[](PJRT_Buffer_GetMemoryLayout_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_GetMemoryLayout");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    return buffer->GetMemoryLayout(args);
  };
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnsafePointer");
    return nullptr;
  };
}

PJRT_Error *
BufferInstance::GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::GetMemoryLayout");
  args->layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  size_t rank = num_dims();
  minor_to_major_.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    minor_to_major_[i] = rank - 1 - i;
  }
  tile_dim_sizes_.resize(1);
  tile_dim_sizes_[0] = rank;
  tile_dims_.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    tile_dims_[i] = dims()[i];
  }
  args->layout.tiled.minor_to_major_size = rank;
  args->layout.tiled.minor_to_major = minor_to_major_.data();
  args->layout.tiled.tile_dims = tile_dims_.data();
  args->layout.tiled.tile_dim_sizes = tile_dim_sizes_.data();

  args->layout.tiled.num_tiles = 1;
  return nullptr;
}

tt_pjrt_status BufferInstance::GetHostSizeInBytes(size_t *host_size) {
  DLOG_F(LOG_DEBUG, "BufferInstance::GetHostSizeInBytes");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::AsyncDeallocate() {
  DLOG_F(LOG_DEBUG, "BufferInstance::AsyncDeallocate");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::Delete() {
  DLOG_F(LOG_DEBUG, "BufferInstance::Delete");
  is_deleted_ = true;
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::CopyToHost(void *dst, size_t dst_size,
                                          EventInstance **out_done_event) {
  DLOG_F(LOG_DEBUG, "BufferInstance::CopyToHost");

  // This callback simply deletes the `dst_buffer_ready_event`. We could perform
  // this deletion in the `dst_buffer_callback`, but this would result in the
  // callback thread of `dst_buffer_ready_event` detaching from the main thread,
  // potentially resulting in the callback thread outliving the main thread.
  auto copy_done_callback = [](PJRT_Error *error, void *user_data) {
    EventInstance *dst_buffer_ready_event =
        static_cast<EventInstance *>(user_data);
    delete dst_buffer_ready_event;
    delete ErrorInstance::FromError(error);
  };

  DLOG_F(INFO, "Copy to host id: %d", unique_id());
  tt::runtime::memcpy(dst, getTensor());

  EventInstance *copy_done_event = new EventInstance();
  copy_done_event->OnReady(copy_done_callback, nullptr);

  *out_done_event = copy_done_event;
  return tt_pjrt_status::kSuccess;
}

PJRT_Buffer_Type BufferInstance::getRuntimeType() {
  DLOG_F(LOG_DEBUG, "BufferInstance::element_type");
  tt::target::DataType Type = tt::runtime::getTensorDataType(getTensor());
  return tt::pjrt::utils::convertElementTypeToBufferType(Type);
}

} // namespace tt::pjrt

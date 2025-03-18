// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "tt/runtime/runtime.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/pjrt_implementation/event_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance;

class BufferInstance {
public:
  BufferInstance(DeviceInstance &device, tt::runtime::Tensor &tensor,
                 const std::vector<std::uint32_t> &shape,
                 const std::vector<std::uint32_t> &stride,
                 std::pair<tt::target::DataType, size_t> tt_buffer_type);

  BufferInstance(DeviceInstance &device, tt::runtime::Tensor &tensor,
                 const std::vector<std::uint32_t> &shape,
                 const std::vector<std::uint32_t> &stride,
                 std::pair<tt::target::DataType, size_t> tt_buffer_type,
                 std::shared_ptr<void> host_buffer_ptr);
  BufferInstance(DeviceInstance &device);
  ~BufferInstance();
  operator PJRT_Buffer *() { return reinterpret_cast<PJRT_Buffer *>(this); }
  static BufferInstance *Unwrap(PJRT_Buffer *buffer) {
    return reinterpret_cast<BufferInstance *>(buffer);
  }
  static void BindApi(PJRT_Api *api);

  // iree_hal_buffer_view_t* buffer_view() { return buffer_view_.get(); }
  DeviceInstance &device() { return device_; }
  const DeviceInstance &device() const { return device_; }
  tt_pjrt_status AsyncDeallocate();
  tt_pjrt_status Delete();
  bool is_deleted() { return is_deleted_; }
  bool is_on_cpu() {
    // TODO: Plumb through an indication if running on CPU and then implement
    // the hook to get an unsafe pointer (avoids a copy).
    return false;
  }
  const tt::runtime::Tensor &getTensor() const { return tensor_; }

  PJRT_Error *GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args *args);
  // Gets the required host size in bytes to copy to host.
  tt_pjrt_status GetHostSizeInBytes(size_t *host_size);
  tt_pjrt_status CopyToHost(void *dst, size_t dst_size,
                            EventInstance **done_event);

  const int64_t *getRawDimensions() { return dims_.data(); }
  const std::vector<std::uint32_t> getDimensions() const {
    return std::vector<std::uint32_t>(dims_.begin(), dims_.end());
  }
  size_t num_dims() { return dims_.size(); }
  void setType(PJRT_Buffer_Type Type) { DataType = Type; }
  std::optional<PJRT_Buffer_Type> getType() { return DataType; }
  const std::shared_ptr<void> get_host_buffer_ptr() const {
    return host_buffer_ptr_;
  }
  const std::vector<std::uint32_t> get_stride() const { return stride_; }
  std::pair<tt::target::DataType, size_t> get_tt_buffer_type() const {
    return tt_buffer_type_;
  }
  // Get the data type for a tensor through runtime if DataType is not set.
  PJRT_Buffer_Type getRuntimeType();

  int unique_id() const { return unique_id_; }

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
  tt::runtime::Tensor tensor_;
  std::pair<tt::target::DataType, size_t> tt_buffer_type_;

  std::vector<int64_t> minor_to_major_;
  std::vector<int64_t> tile_dims_;
  std::vector<size_t> tile_dim_sizes_;

  // Underlying datatype of tensor.
  std::optional<PJRT_Buffer_Type> DataType;

  // OnReady event - currently not used.
  EventInstance *on_ready_event_;

  // Pointer to the host memory used to create this buffer.
  // If buffer is created
  // on device, the value of this pointer is nullptr. It is necessary to keep
  // track of this memory since the runtime will not clean it, and we need to
  // pass the shared pointer to the runtime.
  std::shared_ptr<void> host_buffer_ptr_;
};

} // namespace tt::pjrt

#endif

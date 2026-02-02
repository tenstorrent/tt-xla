// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

// tt-mlir includes
#include "tt/runtime/runtime.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_

namespace tt::pjrt {

class BufferInstance;
class PjrtTensor;

// Tensor pool which holds all pjrt tensor pointers.
//
// Whenever tensor is constructed, it is moved into tensor pool and whenever it
// is destructed, it is moved out.
class PjrtTensorPool {
public:
  void insert(PjrtTensor *tensor);
  void erase(PjrtTensor *tensor);
  void clear();
  void move_tensors_to_host();
  bool contains(PjrtTensor *tensor);

private:
  std::mutex m_mtx;
  std::unordered_set<PjrtTensor *> m_tensors;
};

namespace TensorPool {

PjrtTensorPool &get() noexcept;
const PjrtTensorPool &getc() noexcept;

void insert(PjrtTensor *tensor);
void erase(PjrtTensor *tensor);
void clear();
void move_tensors_to_host();

} // namespace TensorPool

// PJRT tensor class.
//
// Since PJRT does not operate on tensors but on BufferInstances (where each
// BufferInstance represent single tensor shard), we will use our own tensor
// abstraction.
// Each shard will hold shared pointer (PjrtTensorRef) to this tensor for
// automatic memory management.
// Tensor can be constructed from shards or from an existing tensor.
// Whenever tensor is constructed, it is moved into tensor pool and whenever it
// is destructed, it is moved out.
// Note that tensor must be constructed "in reverse" (from shards), because we
// don't know what the tensor even is, until PJRT provide us BufferInstances
// that we can operate on. For that purpose, init functions are provided.
//
// This abstraction gives us control over host and device tensors and hides
// complexity behind simple APIs.
class PjrtTensor {
  // Prevents direct construction. Use PjrtTensor::init instead.
  struct Private {
    explicit Private() = default;
  };

public:
  static PjrtTensor &init_input_tensor(
      const std::vector<BufferInstance *> &shards,
      const tt::runtime::Device &device,
      const std::optional<const tt::runtime::Layout> &layout,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::unordered_map<std::string, std::string> &strategy);

  static PjrtTensor &create(std::vector<BufferInstance *> shards,
                            tt::runtime::Tensor device_tensor);

public: // Constructors needs to be public for std::shared_ptr.
  PjrtTensor(Private, std::vector<BufferInstance *> shards,
             tt::runtime::Tensor tensor);

  ~PjrtTensor();

  PjrtTensor(const PjrtTensor &other) = delete;
  PjrtTensor &operator=(const PjrtTensor &other) = delete;

  PjrtTensor(PjrtTensor &&other) noexcept = delete;
  PjrtTensor &operator=(PjrtTensor &&other) noexcept = delete;

  std::vector<BufferInstance *> &shards() { return m_shards; };
  const std::vector<BufferInstance *> &shards() const { return m_shards; };

  tt::runtime::Tensor &runtime_tensor() { return m_runtime_tensor; }
  const tt::runtime::Tensor &runtime_tensor() const { return m_runtime_tensor; }

  uint64_t uid() const noexcept { return m_uid; }

  // Returns whether shard is part of this tensor.
  bool has_shard(const BufferInstance *shard) const noexcept {
    return std::find(m_shards.begin(), m_shards.end(), shard) != m_shards.end();
  }

  // Return whether runtime tensor has provided layout.
  bool has_layout(const tt::runtime::Layout &layout) const {
    return tt::runtime::hasLayout(m_runtime_tensor, layout);
  };

  // Changes layout of a runtime tensor.
  void to_layout(const tt::runtime::Device &device,
                 const tt::runtime::Layout &layout) {
    m_runtime_tensor =
        tt::runtime::toLayout(m_runtime_tensor, device, layout,
                              tt::runtime::getTensorRetain(m_runtime_tensor));
  };

  // Removes shard from shards (by setting shard to nullptr).
  void remove_shard(const BufferInstance *shard) noexcept;

  // Moves pjrt tensor to host.
  void move_to_host() noexcept;

private:
  static PjrtTensor &
  init_from_existing(const std::vector<BufferInstance *> &shards,
                     const tt::runtime::Device &device,
                     const std::optional<const tt::runtime::Layout> &layout,
                     const std::vector<std::uint32_t> &mesh_shape);

  static PjrtTensor &
  init_new(std::vector<BufferInstance *> shards,
           const tt::runtime::Device &device,
           const std::optional<const tt::runtime::Layout> &layout,
           const std::vector<std::uint32_t> &mesh_shape,
           const std::unordered_map<std::string, std::string> &strategy);

  // Returns whether all shards share the same runtime tensor.
  static bool have_same_tensor(const std::vector<BufferInstance *> &shards);
  static PjrtTensor &from_shards(const std::vector<BufferInstance *> &shards);
  static uint64_t next_uid();

  // Either returns single or multi-device runtime tensor from shards, depending
  // on the strategy.
  static tt::runtime::Tensor rt_tensor_from_strategy(
      const std::vector<BufferInstance *> &shards,
      const std::unordered_map<std::string, std::string> &strategy,
      const std::vector<std::uint32_t> &mesh_shape);

private: // members
  const uint64_t m_uid{next_uid()};
  std::vector<BufferInstance *> m_shards;
  tt::runtime::Tensor m_runtime_tensor;
};

// Shared pointer to pjrt tensor used by all shards that holds tensor.
//
// NOTE: Always use get() to call pjrt tensor methods from this class
// internally if you are adding new methods, because std::shared_ptr does not
// preserve constness of returned pointer with operator->().
// We have also exposed const and non-const APIs for accessing pjrt tensor,
// which ensures that non-const methods from pjrt tensor (which might modify
// this object) are only called on non-const this object.
// This is needed because if we are calling non-const pjrt tensor method, it
// could modify us, which we don't want.
class PjrtTensorRef {
public:
  PjrtTensorRef() = default;

  PjrtTensorRef(const PjrtTensorRef &other) = delete;
  PjrtTensorRef(PjrtTensorRef &&other) = delete;
  PjrtTensorRef &operator=(const PjrtTensorRef &other) = delete;
  PjrtTensorRef &operator=(PjrtTensorRef &&other) = delete;

  ~PjrtTensorRef() { reset(); }

  const PjrtTensor *get() const noexcept { return m_tensor.get(); }
  PjrtTensor *get() noexcept { return m_tensor.get(); }

  void reset(std::shared_ptr<PjrtTensor> tensor = nullptr,
             BufferInstance *shard = nullptr) noexcept {

    assert(!!tensor == !!shard && "Both must be nullptr or have a value.");
    assert(!tensor || tensor->has_shard(shard));

    if (m_shard != nullptr)
      get()->remove_shard(m_shard);

    m_tensor = std::move(tensor);
    m_shard = shard;
  }

  explicit operator bool() const noexcept {
    return static_cast<bool>(m_tensor);
  }

  bool operator==(const PjrtTensorRef &other) const noexcept {
    return get() == other.get();
  }

  const PjrtTensor *operator->() const noexcept {
    return m_tensor.operator->();
  }
  PjrtTensor *operator->() noexcept { return m_tensor.operator->(); }

  const PjrtTensor &operator*() const noexcept { return *m_tensor; };
  PjrtTensor &operator*() noexcept { return *m_tensor; };

private:
  std::shared_ptr<PjrtTensor> m_tensor;
  BufferInstance *m_shard{nullptr};
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_

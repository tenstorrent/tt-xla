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
#include <string>
#include <vector>

// tt-mlir includes
#include "tt/runtime/runtime.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_

namespace tt::pjrt {

class BufferInstance;

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
  // Prevents direct construction. Use static functions below for
  // initialization.
  struct Private {
    explicit Private() = default;
  };

public:
  // Initializes pjrt tensor from pjrt buffers.
  //
  // If tensor already exists (shards already share same tensor) tensor will be
  // reused, and this will behave like a simple getter. Otherwise, new tensor is
  // created based on provided strategy from executable instance.
  static PjrtTensor &from_pjrt_buffers(
      const std::vector<BufferInstance *> &shards,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::unordered_map<std::string, std::string> &strategy);

  // Creates new pjrt tensor for provided shards from an existing runtime
  // tensor.
  static PjrtTensor &from_runtime_tensor(std::vector<BufferInstance *> shards,
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

  // Changes runtime tensor layout if it differs from provided layout.
  void ensure_layout(const tt::runtime::Device &device,
                     const tt::runtime::Layout &layout);

  // Removes shard from shards (by setting shard to nullptr).
  void remove_shard(const BufferInstance *shard) noexcept;

  // Moves pjrt tensor to host.
  void move_to_host() noexcept;

private:
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

private:
  // Tensor unique identifier. For now, used for debug only.
  const uint64_t m_uid;

  // Tensor shards. Each shard hold pjrt tensor reference to this pjrt tensor.
  std::vector<BufferInstance *> m_shards;

  // Underlying runtime tensor.
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
//
// Example:
// void test_function(const BufferInstance* buf) {
//     // This will be allowed if we store std::shared_ptr<PjrtTensor> in
//     // BufferInstance.
//     buf->pjrtTensor()->move_to_host();
// }
//
// With this simple wrapper, we have tied 'const'ness of BufferInstance and
// PjrtTensor.
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
    assert(m_tensor && "Accessing non-existing PJRT tensor");
    return m_tensor.operator->();
  }

  PjrtTensor *operator->() noexcept {
    assert(m_tensor && "Accessing non-existing PJRT tensor");
    return m_tensor.operator->();
  }

  const PjrtTensor &operator*() const noexcept {
    assert(m_tensor && "Accessing non-existing PJRT tensor");
    return *m_tensor;
  };

  PjrtTensor &operator*() noexcept {
    assert(m_tensor && "Accessing non-existing PJRT tensor");
    return *m_tensor;
  };

private:
  std::shared_ptr<PjrtTensor> m_tensor;
  BufferInstance *m_shard{nullptr};
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TENSOR_H_

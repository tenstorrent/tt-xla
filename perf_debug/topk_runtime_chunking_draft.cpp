// Draft: Runtime chunked topk for large vocabs
// Target file: third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/operations/reduction/topk.cpp
//
// This implementation adds vocab-chunking logic to the topk runtime.
// When the input's last dimension > 32768, the runtime splits into
// power-of-2 chunks, runs multi-core topk on each, and merges.
// This makes a single torch.topk(logits, k=32) on [1, 128256] fast
// by using multi-core topk internally (instead of single-core sort).
//
// STATUS: Draft. Not yet compiled/tested. Needs correct ttnn C++ API
// signatures verified against the actual tt-metal headers.

// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt_stl/span.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace tt::runtime::ttnn::operations::reduction::topk {

static constexpr uint32_t kMaxMulticoreTopkSize = 32768;

static uint32_t nextPowerOf2(uint32_t n) {
  uint32_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

static bool shouldChunkTopk(const ::ttnn::Tensor &in, int8_t dim) {
  auto shape = in.logical_shape();
  int rank = static_cast<int>(shape.rank());
  int normalizedDim = dim < 0 ? rank + dim : dim;
  if (normalizedDim != rank - 1) {
    return false;
  }
  return shape[normalizedDim] > kMaxMulticoreTopkSize;
}

static std::vector<::ttnn::Tensor>
chunkedTopk(const ::ttnn::Tensor &in, uint32_t k, int8_t dim, bool largest,
            bool sorted,
            const std::optional<::ttnn::MemoryConfig> &outputMemoryConfig) {
  auto shape = in.logical_shape();
  int rank = static_cast<int>(shape.rank());
  int lastDim = rank - 1;
  uint32_t vocabSize = shape[lastDim];
  uint32_t numChunks =
      (vocabSize + kMaxMulticoreTopkSize - 1) / kMaxMulticoreTopkSize;
  uint32_t chunkSize = (vocabSize + numChunks - 1) / numChunks;
  uint32_t paddedChunkSize = nextPowerOf2(chunkSize);

  LOG_DEBUG("Chunked topk: vocab={}, chunks={}, chunkSize={}, paddedChunkSize={}",
            vocabSize, numChunks, chunkSize, paddedChunkSize);

  std::vector<::ttnn::Tensor> chunkValues;
  std::vector<::ttnn::Tensor> chunkIndices;

  for (uint32_t i = 0; i < numChunks; ++i) {
    uint32_t start = i * chunkSize;
    uint32_t end = std::min(start + chunkSize, vocabSize);

    // Slice chunk from input.
    ::ttsl::SmallVector<int32_t> begins(rank, 0);
    ::ttsl::SmallVector<int32_t> ends;
    for (int d = 0; d < rank; ++d) {
      ends.push_back(static_cast<int32_t>(shape[d]));
    }
    begins[lastDim] = static_cast<int32_t>(start);
    ends[lastDim] = static_cast<int32_t>(end);
    ::ttsl::SmallVector<int32_t> steps(rank, 1);

    ::ttnn::Tensor chunk = ::ttnn::slice(
        in, ttsl::Span<const int32_t>(begins.data(), begins.size()),
        ttsl::Span<const int32_t>(ends.data(), ends.size()),
        ttsl::Span<const int32_t>(steps.data(), steps.size()),
        outputMemoryConfig);

    // Pad to power-of-2.
    uint32_t actualSize = end - start;
    if (actualSize < paddedChunkSize) {
      // NOTE: verify ttnn::pad API signature — may need PadSpecDim vector
      ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> padding;
      for (int d = 0; d < rank; ++d) {
        uint32_t padAmt = (d == lastDim) ? (paddedChunkSize - actualSize) : 0;
        padding.emplace_back(0, padAmt);
      }
      chunk = ::ttnn::pad(chunk, padding,
                          -std::numeric_limits<float>::infinity(),
                          /*use_multicore=*/true, outputMemoryConfig);
    }

    // Typecast to bf16 (topk requires bf16).
    if (chunk.dtype() != ::ttnn::DataType::BFLOAT16) {
      chunk = ::ttnn::typecast(chunk, ::ttnn::DataType::BFLOAT16);
    }

    // Per-chunk topk.
    auto chunkResult = ::ttnn::topk(chunk, k, dim, largest, sorted, outputMemoryConfig);
    chunkValues.push_back(chunkResult[0]);

    // Index offset.
    ::ttnn::Tensor indices = ::ttnn::typecast(chunkResult[1], ::ttnn::DataType::INT32);
    if (i > 0) {
      // NOTE: verify ttnn::full API signature
      ::ttnn::Tensor offset = ::ttnn::full(
          indices.logical_shape(), static_cast<float>(i * chunkSize),
          ::ttnn::DataType::INT32, ::ttnn::Layout::TILE,
          std::nullopt, outputMemoryConfig, indices.device());
      indices = ::ttnn::add(indices, offset);
    }
    chunkIndices.push_back(indices);
  }

  // Merge all chunks.
  ::ttnn::Tensor allValues = ::ttnn::concat(chunkValues, lastDim);
  ::ttnn::Tensor allIndices = ::ttnn::concat(chunkIndices, lastDim);

  // Final topk to reduce [batch, numChunks*k] → [batch, k].
  uint32_t mergedK = numChunks * k;
  if (mergedK > k) {
    uint32_t paddedMergedK = nextPowerOf2(mergedK);
    if (mergedK < paddedMergedK) {
      ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> padding;
      for (int d = 0; d < rank; ++d) {
        uint32_t padAmt = (d == lastDim) ? (paddedMergedK - mergedK) : 0;
        padding.emplace_back(0, padAmt);
      }
      allValues = ::ttnn::pad(allValues, padding,
                              -std::numeric_limits<float>::infinity(),
                              /*use_multicore=*/true, outputMemoryConfig);
      allIndices = ::ttnn::pad(allIndices, padding, 0.0f,
                               /*use_multicore=*/true, outputMemoryConfig);
    }

    auto finalResult = ::ttnn::topk(allValues, k, dim, largest, sorted, outputMemoryConfig);

    // Map local indices → global vocab indices via gather.
    ::ttnn::Tensor localInds = ::ttnn::typecast(finalResult[1], ::ttnn::DataType::INT32);
    // NOTE: verify ttnn::gather signature (dim position, sparse_grad flag)
    ::ttnn::Tensor globalInds = ::ttnn::gather(
        allIndices, lastDim, localInds, /*sparse_grad=*/false, outputMemoryConfig);

    return {finalResult[0], globalInds};
  }

  return {allValues, allIndices};
}

static void runReductionTopKOp(const ::tt::target::ttnn::TopKOp *op,
                               ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  std::vector<::ttnn::Tensor> outputs;
  if (shouldChunkTopk(in, op->dim())) {
    outputs = chunkedTopk(in, op->k(), op->dim(), op->largest(), op->sorted(),
                          outputMemoryConfig);
  } else {
    outputs = ::ttnn::topk(in, op->k(), op->dim(),
                           /*largest=*/op->largest(),
                           /*sorted=*/op->sorted(),
                           /*memory_config=*/outputMemoryConfig,
                           /*sub_core_grids=*/std::nullopt,
                           /*indices_tensor=*/std::nullopt,
                           /*preallocated_output_tensors=*/std::nullopt);
  }

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}

void run(const ::tt::target::ttnn::TopKOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runReductionTopKOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::reduction::topk

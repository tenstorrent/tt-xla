// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// C++ standard library headers
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "utils/callback_worker.h"

namespace tt::pjrt::utils::tests {

// Helper: a simple callback that sets an atomic flag.
static void setFlagCallback(PJRT_Error *error, void *user_arg) {
  auto *flag = reinterpret_cast<std::atomic<bool> *>(user_arg);
  flag->store(true, std::memory_order_release);
}

TEST(CallbackWorkerUnitTests, BasicExecution) {
  CallbackWorker worker;
  std::atomic<bool> executed{false};

  worker.enqueue(setFlagCallback, &executed, nullptr);

  // Wait for the callback to execute (with timeout to avoid hanging).
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!executed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(executed.load(std::memory_order_acquire));
}

// Callback that appends its sequence number to a shared vector.
struct SequenceContext {
  std::atomic<size_t> count{0};
  static constexpr size_t kMaxItems = 256;
  size_t items[kMaxItems];
};

static void recordSequenceCallback(PJRT_Error *error, void *user_arg) {
  auto *ctx = reinterpret_cast<SequenceContext *>(user_arg);
  // The error pointer carries the sequence number (cast from size_t).
  size_t seq = reinterpret_cast<size_t>(error);
  size_t idx = ctx->count.fetch_add(1, std::memory_order_relaxed);
  if (idx < SequenceContext::kMaxItems) {
    ctx->items[idx] = seq;
  }
}

TEST(CallbackWorkerUnitTests, FIFOOrder) {
  CallbackWorker worker;
  SequenceContext ctx;
  constexpr size_t num_items = 64;

  for (size_t i = 0; i < num_items; ++i) {
    // Abuse the error pointer to carry the sequence number.
    worker.enqueue(recordSequenceCallback, &ctx,
                   reinterpret_cast<PJRT_Error *>(i));
  }

  // Wait for all callbacks.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (ctx.count.load(std::memory_order_acquire) < num_items &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(ctx.count.load(std::memory_order_acquire), num_items);

  // Verify FIFO order (single producer, so order must be preserved).
  for (size_t i = 0; i < num_items; ++i) {
    EXPECT_EQ(ctx.items[i], i);
  }
}

// Callback that records both the error pointer and user_arg for verification.
struct ArgsContext {
  std::atomic<bool> called{false};
  PJRT_Error *received_error = nullptr;
  void *received_user_arg = nullptr;
};

static void recordArgsCallback(PJRT_Error *error, void *user_arg) {
  auto *ctx = reinterpret_cast<ArgsContext *>(user_arg);
  ctx->received_error = error;
  ctx->received_user_arg = user_arg;
  ctx->called.store(true, std::memory_order_release);
}

TEST(CallbackWorkerUnitTests, CorrectArgs) {
  CallbackWorker worker;
  ArgsContext ctx;
  auto *fake_error = reinterpret_cast<PJRT_Error *>(0xDEADBEEF);

  worker.enqueue(recordArgsCallback, &ctx, fake_error);

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!ctx.called.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(ctx.called.load(std::memory_order_acquire));
  EXPECT_EQ(ctx.received_error, fake_error);
  EXPECT_EQ(ctx.received_user_arg, static_cast<void *>(&ctx));
}

// Callback that increments an atomic counter.
static void incrementCallback(PJRT_Error *error, void *user_arg) {
  auto *counter = reinterpret_cast<std::atomic<size_t> *>(user_arg);
  counter->fetch_add(1, std::memory_order_relaxed);
}

TEST(CallbackWorkerUnitTests, ConcurrentProducers) {
  constexpr size_t num_producers = 4;
  constexpr size_t items_per_producer = 500;
  constexpr size_t total = num_producers * items_per_producer;

  CallbackWorker worker;
  std::atomic<size_t> counter{0};

  std::vector<std::thread> producers;
  producers.reserve(num_producers);
  for (size_t p = 0; p < num_producers; ++p) {
    producers.emplace_back([&worker, &counter]() {
      for (size_t i = 0; i < items_per_producer; ++i) {
        worker.enqueue(incrementCallback, &counter, nullptr);
      }
    });
  }

  for (auto &t : producers) {
    t.join();
  }

  // Wait for all callbacks to execute.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (counter.load(std::memory_order_acquire) < total &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(counter.load(std::memory_order_acquire), total);
}

TEST(CallbackWorkerUnitTests, ShutdownDrains) {
  constexpr size_t num_items = 100;
  std::atomic<size_t> counter{0};

  {
    CallbackWorker worker;
    for (size_t i = 0; i < num_items; ++i) {
      worker.enqueue(incrementCallback, &counter, nullptr);
    }
    // Worker destroyed here — should drain all remaining items.
  }

  EXPECT_EQ(counter.load(std::memory_order_acquire), num_items);
}

TEST(CallbackWorkerUnitTests, DifferentThread) {
  CallbackWorker worker;
  std::thread::id callback_thread_id;
  std::atomic<bool> done{false};

  auto callback = [](PJRT_Error *error, void *user_arg) {
    auto *ctx =
        reinterpret_cast<std::pair<std::thread::id *, std::atomic<bool> *> *>(
            user_arg);
    *ctx->first = std::this_thread::get_id();
    ctx->second->store(true, std::memory_order_release);
  };

  std::pair<std::thread::id *, std::atomic<bool> *> ctx{&callback_thread_id,
                                                        &done};
  worker.enqueue(callback, &ctx, nullptr);

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(done.load(std::memory_order_acquire));
  EXPECT_NE(callback_thread_id, std::this_thread::get_id());
}

// Context for a self-re-enqueuing callback chain. Each callback fires the next
// one until the desired depth is reached, at which point it signals completion.
struct ReEnqueueContext {
  CallbackWorker *worker = nullptr;
  std::atomic<int> invocation_count{0};
  std::atomic<bool> done{false};
  static constexpr int kChainLength = 5;
};

static void reEnqueueCallback(PJRT_Error *error, void *user_arg) {
  auto *ctx = reinterpret_cast<ReEnqueueContext *>(user_arg);
  int count = ctx->invocation_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count < ReEnqueueContext::kChainLength) {
    ctx->worker->enqueue(reEnqueueCallback, ctx, nullptr);
  } else {
    ctx->done.store(true, std::memory_order_release);
  }
}

// Verify that a callback can safely enqueue new callbacks on the same worker
// (the worker thread acts as a producer for itself). The full chain of
// re-enqueued callbacks must all execute, and in order.
// I don't expect this to happen in practice, but it is a valid use case.
TEST(CallbackWorkerUnitTests, CallbackReEnqueues) {
  CallbackWorker worker;
  ReEnqueueContext ctx;
  ctx.worker = &worker;

  worker.enqueue(reEnqueueCallback, &ctx, nullptr);

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!ctx.done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(ctx.done.load(std::memory_order_acquire));
  EXPECT_EQ(ctx.invocation_count.load(std::memory_order_acquire),
            ReEnqueueContext::kChainLength);
}

} // namespace tt::pjrt::utils::tests

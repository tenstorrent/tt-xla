// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// C++ standard library headers
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "utils/mpsc_queue.h"

namespace tt::pjrt::utils::tests {

TEST(MPSCQueueUnitTests, Construction) {
  MPSCQueue<int> queue(16);
  EXPECT_EQ(queue.capacity(), 16u);
  EXPECT_TRUE(queue.isEmpty());
}

TEST(MPSCQueueUnitTests, PushPopSingle) {
  MPSCQueue<int> queue(4);
  EXPECT_TRUE(queue.tryPush(42));
  EXPECT_FALSE(queue.isEmpty());

  int value = 0;
  EXPECT_TRUE(queue.tryPop(value));
  EXPECT_EQ(value, 42);
  EXPECT_TRUE(queue.isEmpty());
}

TEST(MPSCQueueUnitTests, FIFOOrder) {
  constexpr size_t count = 8;
  MPSCQueue<int> queue(count);

  for (size_t i = 0; i < count; ++i) {
    int val = static_cast<int>(i);
    EXPECT_TRUE(queue.tryPush(std::move(val)));
  }

  for (size_t i = 0; i < count; ++i) {
    int value = -1;
    EXPECT_TRUE(queue.tryPop(value));
    EXPECT_EQ(value, static_cast<int>(i));
  }
}

TEST(MPSCQueueUnitTests, FullQueue) {
  MPSCQueue<int> queue(4);

  for (int i = 0; i < 4; ++i) {
    int val = i;
    EXPECT_TRUE(queue.tryPush(std::move(val)));
  }

  // Queue is full — push should fail.
  int overflow = 99;
  EXPECT_FALSE(queue.tryPush(std::move(overflow)));
}

TEST(MPSCQueueUnitTests, EmptyQueue) {
  MPSCQueue<int> queue(4);
  int value = -1;
  EXPECT_FALSE(queue.tryPop(value));
  EXPECT_EQ(value, -1);
}

TEST(MPSCQueueUnitTests, WrapAround) {
  MPSCQueue<int> queue(4);

  // Push and pop many more items than the capacity to exercise wrap-around.
  for (int round = 0; round < 100; ++round) {
    for (int i = 0; i < 4; ++i) {
      int val = round * 4 + i;
      EXPECT_TRUE(queue.tryPush(std::move(val)));
    }
    for (int i = 0; i < 4; ++i) {
      int value = -1;
      EXPECT_TRUE(queue.tryPop(value));
      EXPECT_EQ(value, round * 4 + i);
    }
  }
}

TEST(MPSCQueueUnitTests, SingleProducerSingleConsumer) {
  constexpr size_t num_items = 100000;
  MPSCQueue<size_t> queue(1024);
  std::vector<size_t> received;
  received.reserve(num_items);

  std::thread producer([&queue]() {
    for (size_t i = 0; i < num_items; ++i) {
      size_t val = i;
      while (!queue.tryPush(std::move(val))) {
        std::this_thread::yield();
      }
    }
  });

  std::thread consumer([&queue, &received]() {
    while (received.size() < num_items) {
      size_t value;
      if (queue.tryPop(value)) {
        received.push_back(value);
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), num_items);
  for (size_t i = 0; i < num_items; ++i) {
    EXPECT_EQ(received[i], i);
  }
}

TEST(MPSCQueueUnitTests, MultiProducerSingleConsumer) {
  constexpr size_t num_producers = 4;
  constexpr size_t items_per_producer = 25000;
  constexpr size_t total_items = num_producers * items_per_producer;
  MPSCQueue<size_t> queue(1024);

  // Each producer writes values encoded as: producer_id * items_per_producer +
  // sequence. This lets us verify per-producer ordering after collection.
  std::vector<size_t> received;
  received.reserve(total_items);
  std::atomic<bool> done{false};

  std::vector<std::thread> producers;
  producers.reserve(num_producers);
  for (size_t p = 0; p < num_producers; ++p) {
    producers.emplace_back([&queue, p]() {
      for (size_t i = 0; i < items_per_producer; ++i) {
        size_t val = p * items_per_producer + i;
        while (!queue.tryPush(std::move(val))) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::thread consumer([&queue, &received, &done]() {
    while (!done.load(std::memory_order_relaxed) ||
           received.size() < num_producers * items_per_producer) {
      size_t value;
      if (queue.tryPop(value)) {
        received.push_back(value);
      } else {
        std::this_thread::yield();
      }
    }
  });

  for (auto &t : producers) {
    t.join();
  }
  done.store(true, std::memory_order_relaxed);
  consumer.join();

  ASSERT_EQ(received.size(), total_items);

  // Verify all items were received (no duplicates, no missing).
  std::vector<size_t> sorted_received = received;
  std::sort(sorted_received.begin(), sorted_received.end());
  for (size_t i = 0; i < total_items; ++i) {
    EXPECT_EQ(sorted_received[i], i);
  }

  // Verify per-producer ordering is preserved: extract each producer's items
  // in the order they were received, and check they are monotonically
  // increasing.
  std::vector<std::vector<size_t>> per_producer(num_producers);
  for (size_t val : received) {
    size_t producer_id = val / items_per_producer;
    per_producer[producer_id].push_back(val);
  }
  for (size_t p = 0; p < num_producers; ++p) {
    ASSERT_EQ(per_producer[p].size(), items_per_producer);
    for (size_t i = 1; i < per_producer[p].size(); ++i) {
      EXPECT_GT(per_producer[p][i], per_producer[p][i - 1]);
    }
  }
}

TEST(MPSCQueueUnitTests, ManyProducersFewItemsEach) {
  constexpr size_t num_producers = 64;
  constexpr size_t items_per_producer = 100;
  constexpr size_t total_items = num_producers * items_per_producer;
  MPSCQueue<size_t> queue(256);

  // Each producer writes values encoded as: producer_id * items_per_producer +
  // sequence. This lets us verify per-producer ordering after collection.
  std::vector<size_t> received;
  received.reserve(total_items);
  std::atomic<bool> done{false};

  std::vector<std::thread> producers;
  producers.reserve(num_producers);
  for (size_t p = 0; p < num_producers; ++p) {
    producers.emplace_back([&queue, p]() {
      for (size_t i = 0; i < items_per_producer; ++i) {
        size_t val = p * items_per_producer + i;
        while (!queue.tryPush(std::move(val))) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::thread consumer([&queue, &received, &done]() {
    while (!done.load(std::memory_order_relaxed) ||
           received.size() < total_items) {
      size_t value;
      if (queue.tryPop(value)) {
        received.push_back(value);
      } else {
        std::this_thread::yield();
      }
    }
  });

  for (auto &t : producers) {
    t.join();
  }
  done.store(true, std::memory_order_relaxed);
  consumer.join();

  ASSERT_EQ(received.size(), total_items);

  // Verify all items were received (no duplicates, no missing).
  std::vector<size_t> sorted_received = received;
  std::sort(sorted_received.begin(), sorted_received.end());
  for (size_t i = 0; i < total_items; ++i) {
    EXPECT_EQ(sorted_received[i], i);
  }

  // Verify per-producer ordering is preserved: extract each producer's items
  // in the order they were received, and check they are monotonically
  // increasing.
  std::vector<std::vector<size_t>> per_producer(num_producers);
  for (size_t val : received) {
    size_t producer_id = val / items_per_producer;
    per_producer[producer_id].push_back(val);
  }
  for (size_t p = 0; p < num_producers; ++p) {
    ASSERT_EQ(per_producer[p].size(), items_per_producer);
    for (size_t i = 1; i < per_producer[p].size(); ++i) {
      EXPECT_GT(per_producer[p][i], per_producer[p][i - 1]);
    }
  }
}

} // namespace tt::pjrt::utils::tests

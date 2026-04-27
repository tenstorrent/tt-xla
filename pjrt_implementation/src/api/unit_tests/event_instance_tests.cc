// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// C++ standard library headers
#include <atomic>
#include <chrono>
#include <thread>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/error_instance.h"
#include "api/event_instance.h"
#include "utils/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt::tests {

// Spin-wait for an atomic flag with a timeout.
static bool waitForFlag(
    const std::atomic<bool> &flag,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!flag.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
}

// Tests successful creation of event instances.
TEST(EventInstanceUnitTests, createInstance_successCase) {
  auto event = EventInstance::createInstance();
  ASSERT_NE(event, nullptr);
  EXPECT_FALSE(event->isReady());
  EXPECT_FALSE(event->isIndestructible());
}

// Tests casting EventInstance to raw PJRT_Event pointer.
TEST(EventInstanceUnitTests, castToPJRTEvent) {
  auto event = EventInstance::createInstance();
  PJRT_Event *pjrt_event = *event;
  EXPECT_NE(pjrt_event, nullptr);
  EXPECT_EQ(static_cast<void *>(event.get()), static_cast<void *>(pjrt_event));
}

// Tests "unwrapping" raw PJRT_Event pointer back to EventInstance.
// Verifies the unwrapped instance matches the original.
TEST(EventInstanceUnitTests, unwrapPJRTEvent) {
  auto event = EventInstance::createInstance();
  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);
  PJRT_Event *pjrt_event = *event;
  EventInstance *unwrapped = EventInstance::unwrap(pjrt_event);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, event.get());
  EXPECT_TRUE(event->isReady());
}

// Tests marking an event as ready with sucess status.
TEST(EventInstanceUnitTests, markAsReady_successStatus) {
  auto event = EventInstance::createInstance();
  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);
  EXPECT_TRUE(event->isReady());
  EXPECT_EQ(event->getErrorFromStatus(), nullptr);
}

// Tests marking an event as ready with error status.
TEST(EventInstanceUnitTests, markAsReady_errorStatus) {
  auto event = EventInstance::createInstance();
  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kAborted);
  EXPECT_TRUE(event->isReady());
  PJRT_Error *pjrt_error = event->getErrorFromStatus();
  ASSERT_NE(pjrt_error, nullptr);
  EXPECT_EQ(ErrorInstance::unwrap(pjrt_error)->getStatus(),
            tt_pjrt_status::kAborted);
}

// Tests marking an event as indestructible.
TEST(EventInstanceUnitTests, setIndestructible) {
  auto event = EventInstance::createInstance();
  event->setIndestructible();
  EXPECT_TRUE(event->isIndestructible());
}

// Tests the PJRT API for checking whether the event is ready.
TEST(EventInstanceUnitTests, API_PJRT_Event_IsReady) {
  auto event = EventInstance::createInstance();

  PJRT_Event_IsReady_Args args;
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.is_ready = true; // intentionally initialized to the opposite value

  PJRT_Error *result = internal::onEventIsReady(&args);
  EXPECT_EQ(result, nullptr);
  EXPECT_FALSE(args.is_ready);
}

// Tests the PJRT API for getting the error code for a ready event.
TEST(EventInstanceUnitTests, API_PJRT_Event_Error) {
  auto event = EventInstance::createInstance();

  PJRT_Event_Error_Args args;
  args.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
  args.event = *event;

  EXPECT_THROW(internal::onEventError(&args), std::runtime_error);

  EventInstance::markAsReadyAndCallback(event.get(),
                                        tt_pjrt_status::kDeadlineExceeded);
  PJRT_Error *error = internal::onEventError(&args);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(ErrorInstance::unwrap(error)->getStatus(),
            tt_pjrt_status::kDeadlineExceeded);
}

// Tests that PJRT API to await returns immediately for ready events.
TEST(EventInstanceUnitTests, API_PJRT_Event_Await_ready) {
  auto event = EventInstance::createInstance();
  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);

  PJRT_Event_Await_Args args;
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = *event;

  // must return immediately since event is already ready
  PJRT_Error *error = internal::onEventAwait(&args);
  EXPECT_EQ(error, nullptr);
}

// Tests the PJRT API to await an event that is not immediately ready.
TEST(EventInstanceUnitTests, API_PJRT_Event_Await_notReady) {
  constexpr int dummy_task_duration_ms = 50;
  auto event = EventInstance::createInstance();

  // this thread will mark the event as ready after a short delay
  std::thread signal_thread([&]() {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(dummy_task_duration_ms));
    EventInstance::markAsReadyAndCallback(event.get(),
                                          tt_pjrt_status::kSuccess);
  });

  PJRT_Event_Await_Args args;
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = *event;

  auto start_time = std::chrono::steady_clock::now();
  PJRT_Error *error = internal::onEventAwait(&args); // must block for a while
  auto end_time = std::chrono::steady_clock::now();
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(event->isReady());

  signal_thread.join();
  auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             end_time - start_time)
                             .count();
  EXPECT_GE(elapsed_time_ms, dummy_task_duration_ms);
}

// Tests the PJRT API to register a callback for an already ready event.
TEST(EventInstanceUnitTests, API_PJRT_Event_OnReady_ready) {
  auto event = EventInstance::createInstance();
  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);

  auto callback = [](PJRT_Error *error, void *user_arg) {
    auto *flag = reinterpret_cast<std::atomic<bool> *>(user_arg);
    flag->store(true, std::memory_order_release);
  };

  std::atomic<bool> callback_executed_flag{false};
  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.callback = callback;
  args.user_arg = &callback_executed_flag;

  PJRT_Error *error = internal::onEventOnReady(&args);

  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(waitForFlag(callback_executed_flag));
}

// Tests the PJRT API to register a callback for an event that
// is not immediately ready.
TEST(EventInstanceUnitTests, API_PJRT_Event_OnReady_notReady) {
  auto event = EventInstance::createInstance();

  auto callback = [](PJRT_Error *error, void *user_arg) {
    auto *flag = reinterpret_cast<std::atomic<bool> *>(user_arg);
    flag->store(true, std::memory_order_release);
  };

  PJRT_Event_OnReady_Args args;
  std::atomic<bool> callback_executed_flag{false};
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.callback = callback;
  args.user_arg = &callback_executed_flag;

  PJRT_Error *error = internal::onEventOnReady(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_FALSE(callback_executed_flag);

  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);
  EXPECT_TRUE(waitForFlag(callback_executed_flag));
}

// Tests the happy path of the callback-destroys-event pattern: the on-ready
// callback calls onEventDestroy, but there are no awaiters, so the event is
// cleanly destroyed. Complements
// API_PJRT_Event_Test_Await_Callbacks_Combination which exercises the same
// pattern with awaiters present.
//
// This also verifies the fix for the original deadlock(or at least one of the
// hypotheses for it) - callbacks previously ran synchronously while
// m_ready_mutex was held, so onEventDestroy (which also acquires m_ready_mutex)
// could deadlock. With the fix, callbacks are dispatched to a dedicated
// CallbackWorker thread after the lock is released.
TEST(EventInstanceUnitTests,
     API_PJRT_Event_OnReady_Callback_Destroys_Event_NoAwaiters) {
  struct CallbackArgs {
    EventInstance *event;
    std::atomic<bool> *callback_ran;
  };

  std::atomic<bool> callback_ran{false};
  auto event = EventInstance::createInstance().release();
  CallbackArgs callback_args{event, &callback_ran};

  auto callback_calls_destroy = [](PJRT_Error *error, void *user_arg) {
    auto *args = reinterpret_cast<CallbackArgs *>(user_arg);
    // Capture the event pointer before signaling, so the test can safely
    // return (taking CallbackArgs off the stack) while we still call destroy.
    EventInstance *event_ptr = args->event;
    args->callback_ran->store(true, std::memory_order_release);
    PJRT_Event_Destroy_Args destroy_args = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
        .event = reinterpret_cast<PJRT_Event *>(event_ptr),
    };
    internal::onEventDestroy(&destroy_args);
  };

  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.callback = callback_calls_destroy;
  args.user_arg = &callback_args;
  PJRT_Error *error = internal::onEventOnReady(&args);
  ASSERT_EQ(error, nullptr);

  EventInstance::markAsReadyAndCallback(event, tt_pjrt_status::kSuccess);

  // Wait for the worker thread to execute the callback.
  EXPECT_TRUE(waitForFlag(callback_ran));
}

// Tests that destroying an event while awaiters are still blocked triggers
// std::terminate(). This is the safety net in EventInstance's destructor: if a
// caller destroys an event with outstanding awaiters, the process crashes
// rather than leaving them waiting on a destroyed condition variable.
//
// The callback-destroys-event pattern (which is how XLA triggers this in
// practice) is already covered by
// API_PJRT_Event_OnReady_Callback_Destroys_Event_NoAwaiters. Here we focus
// purely on the awaiters-still-present invariant.
//
// To make the test deterministic we never mark the event as ready, so awaiters
// remain permanently blocked on the condition variable and m_awaiters_count is
// guaranteed > 0 when the destructor runs.
TEST(EventInstanceUnitTests, API_PJRT_Event_Destroy_With_Active_Awaiters) {
  auto test = []() {
    auto event = EventInstance::createInstance().release();

    constexpr size_t num_waiter_threads = 4;
    std::atomic<size_t> threads_entered{0};

    std::vector<std::thread> waiter_threads;
    waiter_threads.reserve(num_waiter_threads);
    for (size_t i = 0; i < num_waiter_threads; ++i) {
      waiter_threads.emplace_back([&, event] {
        threads_entered.fetch_add(1, std::memory_order_release);
        PJRT_Event_Await_Args await_args;
        await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        await_args.event = reinterpret_cast<PJRT_Event *>(event);
        internal::onEventAwait(&await_args);
      });
    }

    // Wait until all waiter threads have started.
    while (threads_entered.load(std::memory_order_acquire) <
           num_waiter_threads) {
      std::this_thread::yield();
    }
    // Give the threads time to enter await() and block on the condition
    // variable.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Destroy the event while awaiters are still blocked. The destructor
    // should detect m_awaiters_count > 0 and call std::terminate().
    PJRT_Event_Destroy_Args destroy_args = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
        .event = reinterpret_cast<PJRT_Event *>(event),
    };
    internal::onEventDestroy(&destroy_args);

    // Unreachable — std::terminate() is called in the destructor.
    for (auto &thread : waiter_threads) {
      thread.join();
    }
  };
  ASSERT_DEATH(test(), ".*Destroying the event while there are still consumers "
                       "waiting on it!.*");
}

} // namespace tt::pjrt::tests

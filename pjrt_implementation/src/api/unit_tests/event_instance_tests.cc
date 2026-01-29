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
    bool *flag = reinterpret_cast<bool *>(user_arg);
    *flag = true;
  };

  bool callback_executed_flag = false;
  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.callback = callback;
  args.user_arg = &callback_executed_flag;

  PJRT_Error *error = internal::onEventOnReady(&args);

  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(callback_executed_flag);
}

// Tests the PJRT API to register a callback for an event that
// is not immediately ready.
TEST(EventInstanceUnitTests, API_PJRT_Event_OnReady_notReady) {
  auto event = EventInstance::createInstance();

  auto callback = [](PJRT_Error *error, void *user_arg) {
    bool *flag = reinterpret_cast<bool *>(user_arg);
    *flag = true;
  };

  PJRT_Event_OnReady_Args args;
  bool callback_executed_flag = false;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.event = *event;
  args.callback = callback;
  args.user_arg = &callback_executed_flag;

  PJRT_Error *error = internal::onEventOnReady(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_FALSE(callback_executed_flag);

  EventInstance::markAsReadyAndCallback(event.get(), tt_pjrt_status::kSuccess);
  EXPECT_TRUE(callback_executed_flag);
}

// Tests scenario where there are multiple awaiters waiting on an event to be
// ready, but there is also an on-ready callback which destroys the event as
// soon as it is marked ready. Destroying the event will trigger the execution
// of an event destructor, which should terminate the execution since there are
// still awaiters waiting on the event.
TEST(EventInstanceUnitTests, API_PJRT_Event_Test_Await_Callbacks_Combination) {
  auto test = []() {
    auto event = EventInstance::createInstance().release();

    // Create a callback that will be called once the event is ready
    // and which executes event destroy API (which we've observed
    // happens in XLA).
    auto callback_calls_destroy = [](PJRT_Error *error, void *user_arg) {
      PJRT_Event *event = reinterpret_cast<PJRT_Event *>(user_arg);
      PJRT_Event_Destroy_Args destroy_args = {
          .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
          .event = event,
      };
      internal::onEventDestroy(&destroy_args);
    };

    PJRT_Event_OnReady_Args args;
    args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    args.event = *event;
    args.callback = callback_calls_destroy;
    args.user_arg = event;
    PJRT_Error *error = internal::onEventOnReady(&args);
    ASSERT_EQ(error, nullptr);

    // Create waiter threads which will await on event being marked as
    // ready.
    std::vector<std::thread> waiter_threads;
    constexpr size_t num_waiter_threads = 100;
    waiter_threads.reserve(num_waiter_threads);
    for (size_t thread_id = 0; thread_id < num_waiter_threads; ++thread_id) {
      waiter_threads.emplace_back([=] {
        PJRT_Event_Await_Args await_args;
        await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        await_args.event = reinterpret_cast<PJRT_Event *>(event);
        PJRT_Error *error = internal::onEventAwait(&await_args);
        ASSERT_EQ(error, nullptr);
      });
    }

    // Spawn a "worker" thread which will wait and then mark the event
    // as ready. This will trigger callback execution in the same
    // thread.
    std::thread work_thread = std::thread([&]() {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cerr << "Worker thread marking event as ready and executing "
                   "callbacks...\n";
      EventInstance::markAsReadyAndCallback(event, tt_pjrt_status::kSuccess);
    });

    // Wait for all threads to finish.
    work_thread.join();
    for (auto &thread : waiter_threads) {
      thread.join();
    }
  };
  ASSERT_DEATH(
      test(),
      ".*Destroying the event while there are still awaiters waiting on it!*");
}

} // namespace tt::pjrt::tests

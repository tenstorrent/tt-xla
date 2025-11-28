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
  event->markAsReady(tt_pjrt_status::kSuccess); // do something with the event
  PJRT_Event *pjrt_event = *event;
  EventInstance *unwrapped = EventInstance::unwrap(pjrt_event);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, event.get());
  EXPECT_TRUE(event->isReady());
}

// Tests marking an event as ready with sucess status.
TEST(EventInstanceUnitTests, markAsReady_successStatus) {
  auto event = EventInstance::createInstance();
  event->markAsReady(tt_pjrt_status::kSuccess);
  EXPECT_TRUE(event->isReady());
  EXPECT_EQ(event->getErrorFromStatus(), nullptr);
}

// Tests marking an event as ready with error status.
TEST(EventInstanceUnitTests, markAsReady_errorStatus) {
  auto event = EventInstance::createInstance();
  event->markAsReady(tt_pjrt_status::kAborted);
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
  args.is_ready = true; // intentionally initialize to the opposite value

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

  event->markAsReady(tt_pjrt_status::kDeadlineExceeded);
  PJRT_Error *error = internal::onEventError(&args);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(ErrorInstance::unwrap(error)->getStatus(),
            tt_pjrt_status::kDeadlineExceeded);
}

// Tests that PJRT API to await returns immediately for ready events.
TEST(EventInstanceUnitTests, API_PJRT_Event_Await_ready) {
  auto event = EventInstance::createInstance();
  event->markAsReady(tt_pjrt_status::kSuccess);

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
    event->markAsReady(tt_pjrt_status::kSuccess);
  });

  PJRT_Event_Await_Args args;
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = *event;

  auto start_time = std::chrono::steady_clock::now();
  PJRT_Error *error = internal::onEventAwait(&args); // must block for a while
  auto end_time = std::chrono::steady_clock::now();
  EXPECT_EQ(error, nullptr);
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
  event->markAsReady(tt_pjrt_status::kSuccess);

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
TEST(EventInstanceUnitTests, DISABLED_API_PJRT_Event_OnReady_notReady) {
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

  event->markAsReady(tt_pjrt_status::kSuccess);
  event->await();
  EXPECT_TRUE(callback_executed_flag);
}

} // namespace tt::pjrt::tests

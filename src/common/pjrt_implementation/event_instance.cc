// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/event_instance.h"

#include "common/pjrt_implementation/error_instance.h"

namespace tt::pjrt {

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

EventInstance::EventInstance() {
  bool fence = false;
  // TODO: fence and wait
  if (!fence) {
    is_ready_ = true;
    return;
  }

  // {
  //   std::lock_guard<std::mutex> guard(lock_);
  //   // Create a thread that waits on the fence and executes the callbacks
  //   when
  //   // the fence is ready.
  //   signal_thread_ = std::make_unique<std::thread>(
  //       [](EventInstance* event_instance,
  //          iree::vm::ref<iree_hal_fence_t> fence) {
  //         iree_status_t wait_status =
  //             iree_hal_fence_wait(fence.get(), iree_infinite_timeout());
  //         event_instance->SignalReady(wait_status);
  //       },
  //       this, std::move(fence));
  // }
}

EventInstance::~EventInstance() {
  std::lock_guard<std::mutex> guard(lock_);
  if (signal_thread_) {
    if (std::this_thread::get_id() != signal_thread_->get_id()) {
      signal_thread_->join();
    } else {
      // An `EventInstance` is allowed to delete itself in one of its callbacks,
      // resulting in `signal_thread_` being the thread calling the destructor.
      // In such cases, we must let the thread continue running independent of
      // the destructor to avoid a deadlock.
      signal_thread_->detach();
      signal_thread_.release();
    }
  }
}

void EventInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "EventInstance::BindApi");
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Destroy");
    EventInstance *instance = EventInstance::Unwrap(args->event);
    auto delete_event = [](PJRT_Error *error, void *user_data) {
      EventInstance *event = static_cast<EventInstance *>(user_data);
      delete event;
      if (error) {
        delete ErrorInstance::FromError(error);
      }
    };

    instance->OnReady(delete_event, args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_IsReady");
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Error");
    return (PJRT_Error *)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_OnReady");
    return ErrorInstance::MakeError(
        EventInstance::Unwrap(args->event)
            ->OnReady(args->callback, args->user_arg));
  };
}

ErrorInstance *EventInstance::error() {
  std::lock_guard<std::mutex> guard(lock_);
  if (!tt_pjrt_status_is_ok(status_))
    return new ErrorInstance(status_);
  return nullptr;
}
bool EventInstance::is_ready() {
  DLOG_F(LOG_DEBUG, "EventInstance::is_ready");
  std::lock_guard<std::mutex> guard(lock_);
  return is_ready_;
}

tt_pjrt_status EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                      void *user_arg) {
  DLOG_F(LOG_DEBUG, "EventInstance::OnReady");
  tt_pjrt_status local_status;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (!is_ready_) {
      pending_callbacks_.push_back({callback, user_arg});
      return tt_pjrt_status::kSuccess;
    }
    local_status = status_;
  }

  // Already signalled. Callback out of lock scope.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  callback(tt_pjrt_status_is_ok(local_status)
               ? nullptr
               : (PJRT_Error *)new ErrorInstance(local_status),
           user_arg);
  return tt_pjrt_status::kSuccess;
}

void EventInstance::SignalReady(tt_pjrt_status status) {
  DLOG_F(LOG_DEBUG, "EventInstance::SignalReady");
  tt_pjrt_status local_status;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void *>> local_callbacks;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (is_ready_) {
      return;
    }
    local_callbacks.swap(pending_callbacks_);
    is_ready_ = true;
    status_ = status;
    local_status = status_;
  }

  // Trigger callbacks outside of the lock.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  for (auto &cb : local_callbacks) {
    cb.first(tt_pjrt_status_is_ok(local_status)
                 ? nullptr
                 : (PJRT_Error *)new ErrorInstance(local_status),
             cb.second);
  }
}

} // namespace tt::pjrt

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "utils/status.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_EVENT_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_EVENT_INSTANCE_H_

namespace tt::pjrt {

// Represents callback to be executed once the event is ready, each callback
// paired with the pointer to the caller's object that needs to be passed to the
// callback.
struct OnReadyCallback {
  // Callback function to be executed once the event is ready.
  PJRT_Event_OnReadyCallback callback_function;

  // Pointer to the caller's object that needs to be passed as an argument to
  // the callback.
  void *user_arg;
};

// Represents a notifying event that is returned by PJRT APIs that enqueue
// asynchronous work, informing callers when the work is complete and reporting
// a value of type `PJRT_Error*` or `nullptr` as error status.
class EventInstance {
public:
  // Creates new event instance.
  static std::unique_ptr<EventInstance> createInstance();

  // Destructor, handles shutting down the callbacks thread.
  ~EventInstance();

  // Binds PJRT API functions implementation related to PJRT_Event structure.
  static void bindApi(PJRT_Api *api);

  // Casts this event instance to PJRT_Event pointer.
  operator PJRT_Event *() { return reinterpret_cast<PJRT_Event *>(this); }

  // Casts the PJRT_Event pointer to EventInstance pointer.
  static EventInstance *unwrap(PJRT_Event *event) {
    return reinterpret_cast<EventInstance *>(event);
  }

  // Returns true if the event is marked as ready, false otherwise.
  bool isReady();

  // Returns the PJRT_Error created from the event status (nullptr in case of
  // success).
  PJRT_Error *getErrorFromStatus();

  // Marks event as ready with the status of the work. It will cause all
  // callbacks in the callbacks thread to be executed, and will unlock caller's
  // threads awaiting on the event.
  void markAsReady(tt_pjrt_status status);

  // Waits until the event is ready, blocking the calling thread.
  void await();

  // Invokes the callback immediately on the calling thread if the event is
  // ready, otherwise adds it to the list so it can be executed on a separate
  // thread once the event is ready.
  void onReady(PJRT_Event_OnReadyCallback callback_function, void *user_arg);

  // See comment below for `m_indestructible`.
  void setIndestructible() { m_indestructible = true; }
  bool isIndestructible() const { return m_indestructible; }

private:
  // Constructor, spawns the callbacks thread. Private because we wan't events
  // to be created via factory method.
  EventInstance();

  // Kills the callbacks thread;
  void killTheCallbacksThread();

  // True if the event is marked as ready, false otherwise.
  bool m_ready;

  // Status of the work covered by the event.
  tt_pjrt_status m_status;

  // Mutex guarding event state changes.
  std::mutex m_ready_mutex;

  // Condition variable signalling to the waiting threads when event is marked
  // as ready.
  std::condition_variable m_ready_condition;

  // Thread waiting for event to be ready in order to execute the callbacks.
  std::unique_ptr<std::thread> m_callbacks_thread;

  // Holds callbacks to be executed once the event is ready. PJRT docs don't
  // specify if only one callback can be registered per event, so we allow
  // registering multiple.
  std::vector<OnReadyCallback> m_on_ready_callbacks;

  // TODO(mrakita): This is a major hack that we currently have to do because
  // XLA PJRT client destroys event immediately after it sets callback on it.
  // https://github.com/openxla/xla/issues/25172
  bool m_indestructible;
};

namespace internal {

// Implements PJRT_Event_Destroy API function.
PJRT_Error *onEventDestroy(PJRT_Event_Destroy_Args *args);

// Implements PJRT_Event_IsReady API function.
PJRT_Error *onEventIsReady(PJRT_Event_IsReady_Args *args);

// Implements PJRT_Event_Error API function.
PJRT_Error *onEventError(PJRT_Event_Error_Args *args);

// Implements PJRT_Event_Await API function.
PJRT_Error *onEventAwait(PJRT_Event_Await_Args *args);

// Implements PJRT_Event_OnReady API function.
PJRT_Error *onEventOnReady(PJRT_Event_OnReady_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_EVENT_INSTANCE_H_

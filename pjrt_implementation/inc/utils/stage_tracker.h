// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <string>

namespace tt::pjrt::utils {

/**
 * Singleton class for tracking the current pipeline stage.
 * Thread-safe implementation for use across compilation and execution phases.
 *
 * Stage names tracked:
 * - "NOT_STARTED" - Initial state before any processing
 * - "FE_COMPILATION_START" - Frontend compilation (SHLO processing) started
 * - "TTMLIR_COMPILATION_START" - TT-MLIR compilation started
 * - "RUNTIME_EXECUTION_START" - Runtime execution started
 * - "PCC_COMPARISON_START" - PCC comparison started (runtime passed)
 * - "PCC_COMPARISON_PASSED" - PCC comparison passed (model fully passed)
 */
class StageTracker {
public:
  /**
   * Get the singleton instance of StageTracker.
   */
  static StageTracker &getInstance();

  /**
   * Set the current pipeline stage by name.
   * Thread-safe operation.
   */
  void setStage(const char *stage_name);

  /**
   * Get the current pipeline stage as a C string.
   * Returns a pointer to internal storage that remains valid until next setStage().
   * Thread-safe operation.
   */
  const char *getCurrentStageString() const;

  // Delete copy constructor and assignment operator
  StageTracker(const StageTracker &) = delete;
  StageTracker &operator=(const StageTracker &) = delete;

private:
  StageTracker() : current_stage_("NOT_STARTED") {}

  mutable std::mutex mutex_; // Protects current_stage_
  std::string current_stage_;
};

} // namespace tt::pjrt::utils

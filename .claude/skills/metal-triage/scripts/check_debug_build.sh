#!/usr/bin/env bash
# Verify the working tree has the Debug-mode patch applied to
# third_party/CMakeLists.txt that this skill depends on.
#
# We check via `git diff` (not by inspecting the file directly) because the
# checked-in file uses placeholder vars (${TTMLIR_BUILD_TYPE}, ${TT_RUNTIME_DEBUG})
# that are only forced to Debug/ON when the developer has applied the patch.
# Looking at the diff is the most direct way to confirm the override is present.

set -u

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${REPO_ROOT}" ]; then
  echo "Aborting: not inside a git checkout." >&2
  exit 2
fi

DIFF="$(git -C "${REPO_ROOT}" diff -- third_party/CMakeLists.txt)"

needs() {
  local pattern="$1"
  printf '%s' "${DIFF}" | grep -qE "^\+[[:space:]]*${pattern}"
}

MISSING=()
needs '-DCMAKE_BUILD_TYPE=Debug'  || MISSING+=('-DCMAKE_BUILD_TYPE=Debug')
needs '-DTT_RUNTIME_DEBUG=ON'      || MISSING+=('-DTT_RUNTIME_DEBUG=ON')

if [ "${#MISSING[@]}" -ne 0 ]; then
  cat >&2 <<EOF
Aborting: tt-xla is not built in Debug mode.

This skill does NOT build the env. The runtime-debug log lines
("RuntimeTTNN | DEBUG | Executing operation: ...") that the triage workflow
extracts only show up when tt-mlir is built with both:

  -DCMAKE_BUILD_TYPE=Debug
  -DTT_RUNTIME_DEBUG=ON

Neither override is present in your current git diff of
third_party/CMakeLists.txt. Missing: ${MISSING[*]}

Apply the patch to the tt-mlir ExternalProject_Add CMAKE_ARGS in
third_party/CMakeLists.txt (replace \${TTMLIR_BUILD_TYPE} with Debug and
\${TT_RUNTIME_DEBUG} with ON), then rebuild:

  cmake --build build

Then re-invoke the skill.
EOF
  exit 1
fi

echo "ok: Debug build overrides present in third_party/CMakeLists.txt diff."
exit 0

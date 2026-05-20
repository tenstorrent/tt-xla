# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# All docs work is driven by docs/Makefile. These CMake targets are
# thin wrappers so `cmake --build build -- docs` keeps working.
# Assumes docs deps are installed — see docs/build_docs.sh for the
# install + build path used by CI and first-time local builds.

add_custom_target(docs
  COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} html
  USES_TERMINAL
  COMMENT "Building Sphinx documentation"
)

add_custom_target(docs-clean
  COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} clean
  USES_TERMINAL
  COMMENT "Cleaning Sphinx documentation build"
)

add_custom_target(docs-serve
  COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} server
  USES_TERMINAL
  COMMENT "Serving Sphinx documentation locally"
)

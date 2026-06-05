#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Source-able definitions of the representative single-layer perf-test
# subsets. This file is the source of truth for *which* tests run; the
# rationale tables in README.md explain *why* they're chosen.
#
# Sourced by:
#   tt-xla/tests/benchmark/single_layer/regen.sh  (via SUBSET=single|llmbox|galaxy)
#
# The subsets are NON-OVERLAPPING. Each defines just the tests unique to its
# tier:
#   single  = single-chip tests only (8 LLM/encoder, 1 device).
#   llmbox  = TP tests for an 8-device 1×8 mesh.
#   galaxy  = TP tests for a 32-device 4×8 mesh.
#
# The wrapper script accepts a comma-separated list, e.g. SUBSET=single,llmbox
# to run both on an 8-device host, or SUBSET=single,galaxy on a 32-device host.
# Galaxy and llmbox should not be combined: llmbox tests are pinned to 1×8
# mesh configurations and don't fit on 4×8.
#
# These names are passed to the runner via --test (exact, case-insensitive
# match on the model name). Each entry must be a complete model name — no
# prefix subsumption. List every test you want explicitly.

SUBSET_SINGLE="llama_3_2_1b,llama_3_1_8b,phi2,falcon3_1b,gemma_1_1_2b,mistral_7b,qwen_3_0_6b,bert"
SUBSET_LLMBOX="llama_3_1_8b_instruct_tp,falcon3_7b_tp,ministral_8b_tp,gpt_oss_20b_tp,llama_3_1_70b_tp"
SUBSET_GALAXY="llama_3_1_70b_tp_galaxy,gpt_oss_120b_tp_dp_galaxy_batch_size_128"

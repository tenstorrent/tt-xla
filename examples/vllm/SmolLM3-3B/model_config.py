# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SmolLM3-3B model configuration for tt-inference-server integration.

This file documents the intended configuration for deploying SmolLM3-3B
via tt-inference-server. Copy the SMOLLM3_3B_INSTRUCT config into
tt-inference-server's model_config.py when integrating.
"""

# Example tt-inference-server ModelConfig entry:
#
# SMOLLM3_3B_INSTRUCT = ModelConfig(
#     model_name="SmolLM3-3B-Instruct",
#     hf_repo="HuggingFaceTB/SmolLM3-3B",
#     tt_metal_commit="<pin to tested commit>",
#     vllm_commit="<pin to tested vLLM fork commit>",
#     backend="tt_forge",          # set to "tt_metal" until forge path is green
#     supported_devices=["n150", "n300", "p150"],
#     max_seq_len=8192,
#     tensor_parallel=1,
#     dtype="bfloat16",
# )

# Deployment commands:
#
# Single n300 — development iteration:
#   python3 run.py --model SmolLM3-3B-Instruct --device n300 --workflow server --docker-server
#
# Benchmarks against running server:
#   python3 run.py --model SmolLM3-3B-Instruct --device n300 --workflow benchmarks
#
# Full release certification:
#   python3 run.py --model SmolLM3-3B-Instruct --device n300 --workflow release

# Performance targets (n150 / n300, bfloat16, TP=1):
#   TTFT @ seqlen 128:   < 150 ms (n150), < 100 ms (n300)
#   T/S/U (decode):      > 50 tok/s (n150), > 80 tok/s (n300)
#   T/S @ batch 32:      > 800 tok/s (n150), > 1500 tok/s (n300)

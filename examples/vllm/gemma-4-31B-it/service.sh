# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Serves Gemma-4 31B (instruction-tuned) through the tt-xla vLLM plugin on a
# Tenstorrent BHQB 4-chip host (1x4 mesh).
#
# Prerequisites:
#   1. Accept the model license on https://huggingface.co/google/gemma-4-31B-it
#      and authenticate: `huggingface-cli login`.
#   2. Activate the tt-xla environment: `source venv/activate`.
#
# Notes:
#   - max_num_batched_tokens >= 2560 is required because Gemma-4 is multimodal
#     and max_tokens_per_mm_item = 2496 (see vllm/model_executor/models/gemma4_mm.py).
#   - For a different batch size bump both max_num_seqs and
#     max_num_batched_tokens (>= max(batch_size * max_model_len, 2560)) and
#     scale gpu_memory_utilization accordingly.

vllm serve google/gemma-4-31B-it \
    --max-model-len 512 \
    --max-num-batched-tokens 2560 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.1 \
    --additional-config "{\"enable_const_eval\": true, \"min_context_len\": 32, \"enable_tensor_parallel\": true, \"use_2d_mesh\": false, \"cpu_sampling\": false, \"optimization_level\": 0, \"experimental_weight_dtype\": \"bfp_bf8\"}"

#  vllm serve google/gemma-4-31B-it \
#       --max-model-len 512 \
#       --max-num-batched-tokens 2560 \
#       --max-num-seqs 1 \
#       --no-enable-prefix-caching \
#       --gpu-memory-utilization 0.1 \
#       --additional-config '{"enable_const_eval": true, "min_context_len": 32,
#   "enable_tensor_parallel": true, "use_2d_mesh": false, "cpu_sampling": false,
#   "experimental_weight_dtype": "bfp_bf8", "num_hidden_layers": 2,
#   "optimization_level": 2, "debug_dump_logits_dir": "/tmp/logits_opt2",
#   "export_path": "modules/", "export_model_name": "gemma4_2l"}'
#
# --- Debug / bring-up options (add to --additional-config as needed) ---
#
# Cap to 2 transformer layers for fast iteration (removes the full 48-layer stack):
#   \"num_hidden_layers\": 2
#
# Dump per-step device logits to disk for offline comparison between two runs
# (e.g. optimization_level=0 [known-good] vs the failing level):
#   \"debug_dump_logits_dir\": \"/tmp/logits_opt1\"
# Then compare with the companion script:
#   python examples/vllm/gemma-4-31B-it/compare_logits.py /tmp/logits_opt0 /tmp/logits_opt1
#
# Print top-K device tokens live to the server log after every decode step:
#   \"debug_logit_topk\": 5
#
# Export compiled flatbuffers/MLIR artefacts to disk (forwarded to PJRT):
#   \"export_path\": \"/tmp/gemma4_export\"

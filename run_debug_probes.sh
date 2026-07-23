#!/usr/bin/env bash
# DP+TP gemma-4-31B accuracy-bug debug runs.
# Run from repo root with the venv active. Comment out lines you don't want.

mkdir -p debug_logs

PF=tests/integrations/vllm_plugin/generative/test_dp_tp_debug_probes.py
RF=tests/torch/test_dp_tp_fsdp_reshard.py

# =========================================================================
# H2 reshard unit test (cheap; run first)
# =========================================================================
# One param per pytest process: torch_xla's computation client inits once per
# process, and the TT device can't be opened in a forked child (so no --forked).
# Core H2 discriminator is the 8x4 pair; add 4_8 / 2_16 lines if you want the sweep.
tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $RF -k "fsdp_True and 8_4"  2>&1 | tee debug_logs/reshard_fsdpTrue_8x4.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $RF -k "fsdp_False and 8_4" 2>&1 | tee debug_logs/reshard_fsdpFalse_8x4.log

# =========================================================================
# Wave 0 — controls (must pass or nothing downstream is trustworthy)
# =========================================================================
tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_baseline_homogeneous 2>&1 | tee debug_logs/probe_baseline_homogeneous.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_baseline_determinism 2>&1 | tee debug_logs/probe_baseline_determinism.log

# =========================================================================
# Wave 1 — characterize the corruption
# =========================================================================
tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_main 2>&1 | tee debug_logs/probe_main.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_sweep_max_tokens 2>&1 | tee debug_logs/probe_sweep_max_tokens.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_force_full_len 2>&1 | tee debug_logs/probe_force_full_len.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_reorder 2>&1 | tee debug_logs/probe_reorder.log

# =========================================================================
# Wave 2 — localize the code path (TP stays 4)
# =========================================================================
tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_cpu_sampling 2>&1 | tee debug_logs/probe_cpu_sampling.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_const_eval_off 2>&1 | tee debug_logs/probe_const_eval_off.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_shard_weights_off 2>&1 | tee debug_logs/probe_shard_weights_off.log

# =========================================================================
# Wave 3 — confirm the axis (expensive; TP degree also changes)
# =========================================================================
tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_mesh_dp2 2>&1 | tee debug_logs/probe_mesh_dp2.log

tt-smi -r
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $PF -k probe_mesh_dp4 2>&1 | tee debug_logs/probe_mesh_dp4.log

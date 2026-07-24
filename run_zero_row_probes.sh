#!/usr/bin/env bash
# Zero-scheduled-row / early-departure probes. Run from repo root, venv active.
# tt-smi -r before each; DEBUG logging; teed to debug_logs/. Comment out lines
# you don't want. Optionally: export TTXLA_DEBUG_ZERO_ROWS=1  (see
# debug_logs/N1_zero_row_instrumentation.md) to log mid-slice zero rows.

mkdir -p debug_logs
ZF=tests/integrations/vllm_plugin/generative/test_dp_tp_zero_row_probes.py

# Log every condense move with src/dst replica + cross_replica flag (all probes
# here exercise early-departure/condense; harmless where none fire).
export TTXLA_DEBUG_CONDENSE=1

# =========================================================================
# MINIMAL single cross-replica move (cheap Qwen; cleanest correlation)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_minimal_pair 2>&1 | tee debug_logs/zr_minimal_pair.log

# =========================================================================
# N0/N2 — small model first (cheap; enables DP-only comparison)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_small_repro   2>&1 | tee debug_logs/zr_small_repro.log

# DP-degree sweep (dp = 8 / 16 / 32, low TP to avoid TP-compile crashes).
# dp=1 is NOT reachable on 32 chips; confirm instead via TTXLA_DEBUG_CONDENSE
# instrumentation (see debug_logs/N1_zero_row_instrumentation.md).
tt-smi -r
pytest -svv $ZF -k probe_small_dp_tp 2>&1 | tee debug_logs/zr_small_dp_tp.log

tt-smi -r
pytest -svv $ZF -k probe_small_dp16  2>&1 | tee debug_logs/zr_small_dp16.log

tt-smi -r
pytest -svv $ZF -k probe_small_dp32  2>&1 | tee debug_logs/zr_small_dp32.log

# =========================================================================
# Baseline reproduction (gemma)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_repro 2>&1 | tee debug_logs/zr_repro.log

# =========================================================================
# T1 — no early departures without ignore_eos (all prompts > 32 tokens)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_all_long 2>&1 | tee debug_logs/zr_all_long.log

# =========================================================================
# N4 — synchronized early EOS (all finish ~same step)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_sync_short 2>&1 | tee debug_logs/zr_sync_short.log

# =========================================================================
# T2 — protect departers vs protect survivors (per-request ignore_eos)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_protect_departers 2>&1 | tee debug_logs/zr_protect_departers.log

tt-smi -r
pytest -svv $ZF -k probe_protect_survivors 2>&1 | tee debug_logs/zr_protect_survivors.log

# =========================================================================
# T3 — single early finisher, slot 0 vs last slot
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_inject_short_first 2>&1 | tee debug_logs/zr_inject_short_first.log

tt-smi -r
pytest -svv $ZF -k probe_inject_short_last 2>&1 | tee debug_logs/zr_inject_short_last.log

# =========================================================================
# N3 — force chunked prefill (low max_num_batched_tokens)
# =========================================================================
tt-smi -r
pytest -svv $ZF -k probe_chunked_prefill 2>&1 | tee debug_logs/zr_chunked_prefill.log
